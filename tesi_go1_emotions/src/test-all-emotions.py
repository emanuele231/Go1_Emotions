import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
from deepface import DeepFace

# ============================================================
# CARICAMENTO MODELLO MUJOCO
# ============================================================
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/Go1_Emotions/scene.xml"
print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.iterations = 200

# === CONFIGURAZIONE FARI ===
headlight_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_left')
headlight_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_right')
if headlight_left_id == -1 or headlight_right_id == -1:
    print("[ERRORE] Fari non trovati nel modello.")
    exit(1)

mujoco.mj_forward(model, data)

# ============================================================
# POSE DEFINITE
# ============================================================
sad_pose = np.array([
     0.0,  1.2, -2.4,   # Front Right: estese in avanti
     0.0,  1.2, -2.4,   # Front Left
     0.0,  1.8, -2.7,   # Rear Right: molto piegate (seduto)
     0.0,  1.8, -2.7    # Rear Left
])

ready_pose = np.array([
     0.0,  0.9, -1.5,   # Front Right: posizione neutra in piedi
     0.0,  0.9, -1.5,   # Front Left
     0.0,  0.9, -1.5,   # Rear Right
     0.0,  0.9, -1.5    # Rear Left
])

# Posa felice
happy_pose = np.array([
    0.0,  0.7, -1.0,
    0.0,  0.7, -1.0,
    0.0,  0.9, -1.6,
    0.0,  0.9, -1.6
])

print(f"[INFO] Posa SAD: {sad_pose}")
print(f"[INFO] Posa READY: {ready_pose}")
print(f"[INFO] Posa HAPPY: {happy_pose}")

# ============================================================
# STATO E INTERPOLAZIONE
# ============================================================
current_state = "sad"

# ============================================================
# DURATE TRANSIZIONI (in secondi)
# ============================================================
TRANSITION_DURATIONS = {
    ("sad", "ready"):   2.0,   # Si alza in modo deciso
    ("sad", "happy"):   2.0,   # Caso raro, ma gestito
    ("ready", "sad"):   5.0,   # Si "abbatte" lentamente
    ("ready", "happy"): 1.0,   # Scatto rapido e allegro
    ("happy", "sad"):   5.0,   # Torna a terra lentamente
    ("happy", "ready"): 2.0,   # Si calma gradualmente
}
DEFAULT_TRANSITION = 2.0

transition_duration = DEFAULT_TRANSITION
transition_start_time = None
start_pose = sad_pose.copy()
target_pose = sad_pose.copy()
interpolated_pose = sad_pose.copy()

# ============================================================
# SALTO (HOP) - SOLO PER HAPPY
# ============================================================
hop_active = False
hop_start_time = 0
hop_duration = 1.0
hop_height = 0.08  # Saltino più visibile

height_sad = 0.15       # Robot seduto/sdraiato (DEVE essere basso)
height_ready = 0.32     # Robot in piedi
height_happy = 0.32     # Robot in piedi (base per il salto)

current_height = height_sad
start_height = height_sad      # Altezza di partenza della transizione
target_height = height_sad     # Altezza di arrivo della transizione

KP = 40.0
KD = 0.0

def smoothstep(t):
    return t * t * (3 - 2 * t)

def apply_state(state_name):
    global current_state, start_pose, target_pose, transition_start_time
    global hop_active, hop_start_time, transition_duration
    global start_height, target_height  # ← AGGIUNGI
    
    # Evita transizioni ridondanti
    if current_state == state_name and transition_start_time is None:
        return
    
    # Determina la durata in base alla transizione specifica
    transition_key = (current_state, state_name)
    transition_duration = TRANSITION_DURATIONS.get(transition_key, DEFAULT_TRANSITION)
    print(f"[INFO] Durata transizione {current_state} -> {state_name}: {transition_duration}s")
    
    start_pose = interpolated_pose.copy()
    start_height = current_height  # ← SALVA L'ALTEZZA CORRENTE
    
    if state_name == "sad":
        target_pose = sad_pose.copy()
        target_height = height_sad  # ← ALTEZZA TARGET
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: SAD (fari spenti)")
        hop_active = False
        
    elif state_name == "happy":
        target_pose = happy_pose.copy()
        target_height = height_happy  # ← ALTEZZA TARGET
        model.geom_rgba[headlight_left_id] = [0.0, 1.0, 0.0, 1.0]
        model.geom_rgba[headlight_right_id] = [0.0, 1.0, 0.0, 1.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: HAPPY (fari verdi + saltino)")
        hop_active = True
        hop_start_time = time.time()
        
    elif state_name == "ready":
        target_pose = ready_pose.copy()
        target_height = height_ready  # ← ALTEZZA TARGET
        model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]
        model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: READY (fari bianchi)")
    
    current_state = state_name
    transition_start_time = time.time()

def controller(data, model):
    global interpolated_pose, transition_start_time, hop_active, hop_start_time
    global transition_duration, current_height
    
    # --- INTERPOLAZIONE POSE E ALTEZZA ---
    if transition_start_time is not None:
        elapsed = time.time() - transition_start_time
        t = min(elapsed / transition_duration, 1.0)
        t_smooth = smoothstep(t)
        
        # Interpola i giunti
        interpolated_pose = start_pose + t_smooth * (target_pose - start_pose)
        
        # Interpola l'altezza (stessa curva smoothstep)
        current_height = start_height + t_smooth * (target_height - start_height)
        
        if t >= 1.0:
            transition_start_time = None
    
    # Controllo PD per i giunti
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = interpolated_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)
    
    # SALTO (HOP) SOLO SU ASSE VERTICALE
    if hop_active and hop_start_time is not None:
        elapsed_hop = time.time() - hop_start_time
        
        if elapsed_hop < hop_duration:
            t_hop = elapsed_hop / hop_duration
            hop_offset = hop_height * np.sin(np.pi * t_hop)
            data.qpos[2] = current_height + hop_offset
        else:
            data.qpos[2] = current_height
            hop_active = False
            hop_start_time = None
    else:
        data.qpos[2] = current_height

# ============================================================
# CONFIGURAZIONE WEBCAM + HAAR CASCADE
# ============================================================
cascade_path = "/home/emanuele/unitree_sdk2_python/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("[ERRORE] Impossibile caricare Haar Cascade.")
    exit(1)
        
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)
print("[OK] Webcam attiva")

# === PARAMETRI TEMPORALI ===
PRESENCE_THRESHOLD = 2.5      
ABSENCE_THRESHOLD = 5.0       
VISION_INTERVAL = 0.1         
EMOTION_INTERVAL = 1.5        

last_vision_time = 0
last_emotion_time = 0
face_start_time = None
last_face_seen_time = time.time()

# ============================================================
# SETUP INIZIALE MUJOCO
# ============================================================
apply_state("sad")
data.qpos[2] = current_height
data.qpos[7:19] = interpolated_pose
mujoco.mj_forward(model, data)

print(f"Numero di motori: {model.nu}")
print(f"Altezza iniziale: {data.qpos[2]:.3f} m")
print("\n" + "=" * 50)
print("SISTEMA EMOZIONI COMPLETE:")
print("  - Nessuno -> SAD (fari spenti)")
print("  - Presenza rilevata -> READY (fari bianchi)")
print("  - Sorriso -> HAPPY (fari verdi + saltino)")
print("=" * 50)

# ============================================================
# LOOP PRINCIPALE
# ============================================================
print("\nAvvio simulazione...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    while viewer.is_running():
        step_start = time.perf_counter()
        current_time = time.time()
        
        # --- RILEVAMENTO VOLTO (Haar Cascade, 10Hz) ---
        if current_time - last_vision_time > VISION_INTERVAL:
            last_vision_time = current_time
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_detected = len(faces) > 0
                
                # Aggiorna timer presenza
                if face_detected:
                    last_face_seen_time = current_time
                    if face_start_time is None:
                        face_start_time = current_time
                else:
                    face_start_time = None
                
                # --- TRANSIZIONE A READY QUANDO APPARE IL VOLTO ---
                if face_detected and face_start_time is not None:
                    if (current_time - face_start_time) >= PRESENCE_THRESHOLD:
                        if current_state == "sad":
                            apply_state("ready")
                
                # --- ANALISI EMOZIONE (DeepFace, ogni 1.5s) ---
                if face_detected and (current_time - last_emotion_time > EMOTION_INTERVAL):
                    last_emotion_time = current_time
                    
                    try:
                        # Analizza solo se la presenza è confermata
                        if face_start_time is not None and (current_time - face_start_time) >= PRESENCE_THRESHOLD:
                            result = DeepFace.analyze(
                                frame,
                                actions=['emotion'],
                                enforce_detection=False,
                                silent=True
                            )
                            
                            if isinstance(result, list):
                                result = result[0]
                            
                            dominant = result['dominant_emotion']
                            happy_score = result['emotion'].get('happy', 0.0)
                            
                            # STAMPA SEMPRE VISIBILE DEI VALORI
                            print(f"\n{'='*60}")
                            print(f"[{time.strftime('%H:%M:%S')}] ANALISI EMOZIONE")
                            print(f"  Stato attuale: {current_state.upper()}")
                            print(f"  Emozione dominante: {dominant.upper()}")
                            print(f"  Punteggio HAPPY: {happy_score:.1f}%")
                            print(f"{'='*60}\n")
                            
                            # Se happy > 50% e siamo in ready, passa a happy
                            if happy_score > 50.0 and current_state == "ready":
                                apply_state("happy")
                            # Se happy < 30% e siamo in happy, torna a ready
                            elif happy_score < 30.0 and current_state == "happy":
                                apply_state("ready")
                                
                    except Exception as e:
                        print(f"[ERRORE DeepFace] {e}")
                
                # --- MACCHINA A STATI PRESENZA ---
                # Se non c'è più nessuno, torna a sad
                if current_state != "sad" and face_start_time is None:
                    if (current_time - last_face_seen_time) >= ABSENCE_THRESHOLD:
                        apply_state("sad")
                        face_start_time = None
                
                # --- DEBUG VISIVO WEBCAM ---
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Mostra stato a video
                status_text = f"Stato: {current_state.upper()}"
                if current_state == "happy":
                    color = (0, 255, 0)
                elif current_state == "ready":
                    color = (255, 255, 255)
                else:
                    color = (0, 255, 255)
                
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Webcam - Emotion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # --- CONTROLLO MUJOCO (gira sempre, 500Hz) ---
        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # --- DIAGNOSTICA ---
        if int(data.time / model.opt.timestep) % 1000 == 0:
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m | Stato: {current_state.upper()}")
        
        # --- TIMING LOOP ---
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# ============================================================
# PULIZIA
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("\nSimulazione terminata.")

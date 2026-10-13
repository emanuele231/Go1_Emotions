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

# Default (da usare per ogni emozione in prova)
neutral_pose = np.array([
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5
])

# Posa felice
happy_pose = np.array([
    0.0,  0.7, -1.0,
    0.0,  0.7, -1.0,
    0.0,  0.9, -1.6,
    0.0,  0.9, -1.6
])

print(f"[INFO] Posa NEUTRAL: {neutral_pose}")
print(f"[INFO] Posa HAPPY: {happy_pose}")

current_state = "happy"
transition_duration = 1.0
transition_start_time = None
start_pose = neutral_pose.copy()
target_pose = neutral_pose.copy()
interpolated_pose = neutral_pose.copy()

hop_active = False
hop_start_time = 0
hop_duration = 1.0
hop_height = 0.05
base_height = 0.33


KP = 20.0
KD = 0.0

def smoothstep(t):
    return t * t * (3 - 2 * t)

def apply_state(state_name):
    global current_state, start_pose, target_pose, transition_start_time
    global hop_active, hop_start_time
    
    # Evita transizioni ridondanti
    if current_state == state_name and transition_start_time is None:
        return
    
    start_pose = interpolated_pose.copy()
    
    if state_name == "neutral":
        target_pose = neutral_pose.copy()
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: NEUTRAL (fari spenti)")
        hop_active = False
        
    elif state_name == "happy":
        target_pose = happy_pose.copy()
        model.geom_rgba[headlight_left_id] = [0.0, 1.0, 0.0, 1.0]
        model.geom_rgba[headlight_right_id] = [0.0, 1.0, 0.0, 1.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: HAPPY (fari verdi)")
        hop_active = True
        hop_start_time = time.time()
    
    current_state = state_name
    transition_start_time = time.time()

def controller(data, model):
    global interpolated_pose, transition_start_time, hop_active, hop_start_time
    
    if transition_start_time is not None:
        elapsed = time.time() - transition_start_time
        t = min(elapsed / transition_duration, 1.0)
        t_smooth = smoothstep(t)
        interpolated_pose = start_pose + t_smooth * (target_pose - start_pose)
        if t >= 1.0:
            transition_start_time = None
    
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = interpolated_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)
        
        # --- SALTO (HOP) SOLO SU ASSE VERTICALE ---
    if hop_active and hop_start_time is not None:
        elapsed_hop = time.time() - hop_start_time
        
        if elapsed_hop < hop_duration:
            t_hop = elapsed_hop / hop_duration
            hop_offset = hop_height * np.sin(np.pi * t_hop)
            data.qpos[2] = base_height + hop_offset
        else:
            data.qpos[2] = base_height
            hop_active = False
            hop_start_time = None
    else:
        data.qpos[2] = base_height
    
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
PRESENCE_THRESHOLD = 1.5      
ABSENCE_THRESHOLD = 3.0       
VISION_INTERVAL = 0.1         
EMOTION_INTERVAL = 1.5        # DeepFace ogni 2 secondi (0.5Hz)

last_vision_time = 0
last_emotion_time = 0
face_start_time = None
last_face_seen_time = time.time()

apply_state("neutral")
data.qpos[2] = 0.33
data.qpos[7:19] = interpolated_pose
mujoco.mj_forward(model, data)

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
                
                # --- ANALISI EMOZIONE (DeepFace, ogni 2s) ---
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
                            
                            print(f"[{time.strftime('%H:%M:%S')}] Emozione: {dominant} (happy: {happy_score:.1f}%)")
                            
                            # Se happy > 50% e non siamo già in happy
                            if happy_score > 50.0 and current_state != "happy":
                                apply_state("happy")
                            # Se happy < 30% e siamo in happy
                            elif happy_score < 30.0 and current_state == "happy":
                                apply_state("neutral")
                                
                    except Exception as e:
                        print(f"[ERRORE DeepFace] {e}")
                
                # --- MACCHINA A STATI PRESENZA ---
                # Se non c'è più nessuno, torna a neutral
                if current_state != "neutral" and face_start_time is None:
                    if (current_time - last_face_seen_time) >= ABSENCE_THRESHOLD:
                        apply_state("neutral")
                
                # --- DEBUG VISIVO WEBCAM ---
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Mostra stato a video
                status_text = f"Stato: {current_state.upper()}"
                color = (0, 255, 0) if current_state == "happy" else (0, 255, 255)
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
            
cap.release()
cv2.destroyAllWindows()
print("\nSimulazione terminata.")

        

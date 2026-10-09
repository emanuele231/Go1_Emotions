import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2

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

print(f"[INFO] Posa SAD (neutra): {sad_pose}")
print(f"[INFO] Posa READY (testa alzata): {ready_pose}")


current_state = "sad"
transition_duration = 2.0
transition_start_time = None
start_pose = sad_pose.copy()
target_pose = sad_pose.copy()
interpolated_pose = sad_pose.copy()

KP = 40.0
KD = 0.0

def smoothstep(t):
    return t * t * (3 - 2 * t)

def apply_state(state_name):
    global current_state, start_pose, target_pose, transition_start_time
    
    # Evita transizioni ridondanti
    if current_state == state_name and transition_start_time is None:
        return
    
    start_pose = interpolated_pose.copy()
    
    if state_name == "sad":
        target_pose = sad_pose.copy()
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: SAD (fari spenti)")
        
    elif state_name == "ready":
        target_pose = ready_pose.copy()
        model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]
        model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: READY (fari accesi)")
    
    current_state = state_name
    transition_start_time = time.time()

def controller(data, model):
    global interpolated_pose, transition_start_time
    
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

# === PARAMETRI DEBOUNCE ===
PRESENCE_THRESHOLD = 2.5  # Secondi di presenza continua per "ready"
ABSENCE_THRESHOLD = 5.0   # Secondi di assenza per "sad"
last_vision_time = 0      # Per limitare il rilevamento volto a 10Hz

face_start_time = None
last_face_seen_time = time.time()


apply_state("sad")
data.qpos[2] = 0.32
data.qpos[7:19] = interpolated_pose
mujoco.mj_forward(model, data)

print("\n" + "=" * 50)
print(f"  - Presenza confermata dopo {PRESENCE_THRESHOLD}s -> READY")
print(f"  - Assenza confermata dopo {ABSENCE_THRESHOLD}s -> SAD")
print("  - [ESC] nella finestra MuJoCo per uscire")
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
        
        if current_time - last_vision_time > 0.1:  # 10Hz
            last_vision_time = current_time
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_detected = len(faces) > 0
                
                if face_detected:
                    last_face_seen_time = current_time
                    if face_start_time is None:
                        face_start_time = current_time
                else:
                    face_start_time = None
                
                # --- MACCHINA A STATI ---
                if current_state == "sad" and face_start_time is not None:
                    if (current_time - face_start_time) >= PRESENCE_THRESHOLD:
                        apply_state("ready")
                        
                elif current_state == "ready":
                    if (current_time - last_face_seen_time) >= ABSENCE_THRESHOLD:
                        apply_state("sad")
                        face_start_time = None
                
                # --- DEBUG VISIVO WEBCAM ---
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if current_state == "sad" and face_start_time is not None:
                    tbs = current_time - face_start_time
                    cv2.putText(frame, f"Conferma: {tbs:.1f}s/{PRESENCE_THRESHOLD}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif current_state == "ready":
                    ts = current_time - last_face_seen_time
                    cv2.putText(frame, f"READY | Assenza: {ts:.1f}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "SAD (in attesa di volto)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('Webcam - Face Presence', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # --- CONTROLLO MUJOCO (gira sempre, indipendentemente dal rilevamento) ---
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

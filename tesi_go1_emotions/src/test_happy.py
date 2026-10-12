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

#MOVIMENTO FELICE

print(f"[INFO] Posa HAPPY : {happy_pose}")

current_state = "happy"
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
   
        
    elif state_name == "happy":
        target_pose = ready_pose.copy()
        model.geom_rgba[headlight_left_id] = [0.0, 1.0, 0.0, 1.0]
        model.geom_rgba[headlight_right_id] = [0.0, 1.0, 0.0, 1.0]
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
        
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)
print("[OK] Webcam attiva")

 ============================================================
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

                
                # --- DEBUG VISIVO WEBCAM ---
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
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

        

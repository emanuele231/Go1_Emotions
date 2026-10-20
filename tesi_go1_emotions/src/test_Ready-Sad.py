import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
import urllib.request

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
     0.0,  1.2, -2.5,   # Rear Right: molto piegate (seduto)
     0.0,  1.2, -2.5    # Rear Left
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

# === FACE TRACKING (PROVVISORI) ===
FIXATION_THRESHOLD = 10.0    # secondi di sguardo continuo dopo il READY
WANDER_AMPLITUDE   = 0.3     # radianti (~17.2°) per lato
WANDER_PERIOD      = 3.0     # secondi per un ciclo completo destra-sinistra
WANDER_RAMP        = 1.0     # rampa di ingresso dell'ampiezza (evita strattoni)

YAW_KP             = 2.0     # Nm per radiante di errore
YAW_KD             = 1.0     # Nm per rad/s
YAW_TORQUE_MAX     = 3.0     # saturazione coppia [Nm]

wander_active      = False
wander_start_time  = 0.0
wander_yaw_target  = 0.0

KP = 40.0
KD = 0.0

def smoothstep(t):
    return t * t * (3 - 2 * t)
    
    # === PROVVISORIO ==
def quat_to_yaw(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
    
def update_wander_torque(data, now):
    """Genera il riferimento yaw e applica la coppia sul torace."""
    global wander_yaw_target

    # --- Riferimento: sinusoide continua, con rampa di ingresso ---
    if wander_active:
        t = now - wander_start_time
        ramp = smoothstep(min(t / WANDER_RAMP, 1.0))
        wander_yaw_target = (WANDER_AMPLITUDE * ramp
                             * np.sin(2.0 * np.pi * t / WANDER_PERIOD))
    else:
        wander_yaw_target = 0.0   # ritorno dolce al centro

    # --- Orientazione del torso ---
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    R = R.reshape(3, 3)

    yaw = quat_to_yaw(data.qpos[3:7])
    yaw_err = wrap_angle(wander_yaw_target - yaw)

    # angular velocity del freejoint e' nel frame locale -> mondo
    omega_world = R @ data.qvel[3:6]

    tau = YAW_KP * yaw_err - YAW_KD * omega_world[2]
    tau = float(np.clip(tau, -YAW_TORQUE_MAX, YAW_TORQUE_MAX))

    # coppia attorno all'asse Z mondo, espressa nel frame locale del base
    data.qfrc_applied[3:6] = R.T @ np.array([0.0, 0.0, tau])

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
    
    update_wander_torque(data, current_time)

# ============================================================
# CONFIGURAZIONE WEBCAM + YuNet
# ============================================================

YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

if not os.path.exists(YUNET_MODEL_PATH):
    print(f"[INFO] Download modello YuNet in corso...")
    urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
    print(f"[OK] Modello YuNet scaricato: {YUNET_MODEL_PATH}")
else:
    print(f"[OK] Modello YuNet trovato: {YUNET_MODEL_PATH}")
    
YUNET_SCORE_THRESHOLD = 0.7
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

face_detector = cv2.FaceDetectorYN_create(
    YUNET_MODEL_PATH,
    "",  # config (non serve per ONNX)
    (320, 320),  # dimensione iniziale, verrà aggiornata al primo frame
    score_threshold=YUNET_SCORE_THRESHOLD,
    nms_threshold=YUNET_NMS_THRESHOLD,
    top_k=YUNET_TOP_K
)
print(f"[OK] YuNet inizializzato (score_thr={YUNET_SCORE_THRESHOLD})")

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
                
                h, w, _ = frame.shape
                face_detector.setInputSize((w, h))
                _retval, faces = face_detector.detect(frame)
                
                # `faces` è un array Nx15: [x, y, w, h, x_reye, y_reye, x_leye, y_leye,
                #                            x_nose, y_nose, x_rcorner, y_rcorner,
                #                            x_lcorner, y_lcorner, score]
                face_detected = (faces is not None and len(faces) > 0)
                
                if face_detected:
                    last_face_seen_time = current_time
                    if face_start_time is None:
                        face_start_time = current_time
                else:
                    face_start_time = None
                
                if current_state == "sad" and face_start_time is not None:
                    wander_active = False
                    if (current_time - face_start_time) >= PRESENCE_THRESHOLD:
                        apply_state("ready")
                        
                elif current_state == "ready":
                     if (current_time - last_face_seen_time) >= ABSENCE_THRESHOLD:
                         wander_active = False
                         apply_state("sad")
                         face_start_time = None
                     else:
                        if wander_active and face_start_time is None:
                           wander_active = False
                           print(f"[{time.strftime('%H:%M:%S')}] Sguardo perso -> stop oscillazione")

                        if face_start_time is not None and not wander_active:
                           if (current_time - face_start_time) >= FIXATION_THRESHOLD:
                              wander_active = True
                              wander_start_time = current_time
                              print(f"[{time.strftime('%H:%M:%S')}] Fissazione confermata -> LOOP YAW")
                
                # --- DEBUG VISIVO WEBCAM ---
                
                img_h, img_w, _ = frame.shape
                center_img_x = img_w // 2
                center_img_y = img_h // 2
                
                if wander_active:
                   deg = np.degrees(wander_yaw_target)
                   cv2.putText(frame, f"LOOK AROUND", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
  
                # Disegna croce al centro dell'immagine (riferimento)
                cv2.drawMarker(frame, (center_img_x, center_img_y), 
                              (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
                if face_detected:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    
                    x, y, fw, fh = best_face[0:4].astype(int)
                    score = best_face[-1]
                    
                    cx = x + fw // 2
                    cy = y + fh // 2
                    
                    x_norm = (cx - center_img_x) / center_img_x
                    y_norm = (cy - center_img_y) / center_img_y
                    
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
                    # Landmark (occhi, naso, bocca)
                    landmarks = best_face[4:14].astype(int).reshape(5, 2)
                    for (lx, ly) in landmarks:
                        cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
    
                    # --- Disegno centro volto e linea di offset ---
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.line(frame, (center_img_x, center_img_y), (cx, cy), (0, 255, 255), 1)
    
                    # --- Stampa coordinate a video ---
                    cv2.putText(frame, f"Face center: ({cx}, {cy})", (10, img_h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Normalized:  ({x_norm:+.2f}, {y_norm:+.2f})", (10, img_h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Size:        {fw}x{fh} px", (10, img_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
                    face_data = {
                        'cx': cx, 'cy': cy,
                        'x_norm': x_norm, 'y_norm': y_norm,
                        'size': fw * fh,
                        'detected': True
                    }
                else:
                    face_data = {
                        'cx': center_img_x, 'cy': center_img_y,
                        'x_norm': 0.0, 'y_norm': 0.0,
                        'size': 0,
                        'detected': False
                    }
                
                # --- OVERLAY STATO ---
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
                
                cv2.imshow('Webcam - YuNet Face Presence', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # --- CONTROLLO MUJOCO (gira sempre, indipendentemente dal rilevamento) ---
        data.qfrc_applied[:] = 0.0
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

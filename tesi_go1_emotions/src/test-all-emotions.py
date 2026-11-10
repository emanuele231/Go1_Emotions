import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import math

# ============================================================
# CARICAMENTO MODELLO MUJOCO
# ============================================================
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/Go1_Emotions/scene.xml"
print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.iterations = 200

headlight_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_left')
headlight_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_right')
if headlight_left_id == -1 or headlight_right_id == -1:
    print("[ERRORE] Fari non trovati nel modello.")
    exit(1)

mujoco.mj_forward(model, data)

# ============================================================
# POSE DEFINITE
# ============================================================
neutral_pose = np.array([
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5
])

# ============================================================
# STATO E PARAMETRI
# ============================================================
current_height = 0.32

# Osservazione
current_yaw = 0.0
target_yaw = 0.0

walk_phase = 0.0
last_target_yaw = 0.0
walk_amplitude = 0.0

KP = 20.0
KD = 0.0

VISION_INTERVAL = 0.1
last_vision_time = 0.0

def get_walking_pose(current_time, current_yaw, target_yaw):
    base_pose = neutral_pose.copy()
    
    yaw_error = target_yaw - current_yaw
    
    if abs(yaw_error) < 0.015:
        return base_pose
        
    frequency = 3.5
    phase = current_time * frequency * 2 * math.pi
    amplitude = min(abs(yaw_error) * 3.0, 0.15)
    
    swing = math.sin(phase) * amplitude
    
    base_pose[0] += swing
    base_pose[3] -= swing
    base_pose[6] += swing
    base_pose[9] -= swing
    
    step_bend = math.sin(phase) * amplitude * 0.5
    
    base_pose[1] -= step_bend * 0.5
    base_pose[2] += step_bend * 0.5
    base_pose[4] += step_bend * 0.5
    base_pose[5] -= step_bend * 0.5
    base_pose[7] -= step_bend * 0.5
    base_pose[8] += step_bend * 0.5
    base_pose[10] += step_bend * 0.5
    base_pose[11] -= step_bend * 0.5
    
    return base_pose

def controller(data, model):
    global walk_phase, walk_amplitude, target_yaw

    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    actual_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    yaw_error = target_yaw - actual_yaw
    
    while yaw_error > math.pi: yaw_error -= 2 * math.pi
    while yaw_error < -math.pi: yaw_error += 2 * math.pi

    needs_turning = abs(yaw_error) > 0.05

    if needs_turning:
        data.qvel[5] = math.copysign(0.8, yaw_error)
        
        walk_phase += 0.25 # Velocità del passo
        walk_amplitude = 1.0
    else:
        data.qvel[5] = 0.0
        walk_amplitude += (0.0 - walk_amplitude) * 0.1 # Fade out dolce

    data.qvel[0] = 0.0
    data.qvel[1] = 0.0
    data.qvel[2] = 0.0
    data.qvel[3] = 0.0 # Roll
    data.qvel[4] = 0.0 # Pitch

    dynamic_pose = neutral_pose.copy()
    
    swing_1 = math.sin(walk_phase) * walk_amplitude
    swing_2 = math.sin(walk_phase + math.pi) * walk_amplitude

    dynamic_pose[1] += swing_1 * 1.5
    dynamic_pose[2] -= swing_1 * 1.2
    dynamic_pose[10] += swing_1 * 1.5
    dynamic_pose[11] -= swing_1 * 1.2

    dynamic_pose[4] += swing_2 * 1.5
    dynamic_pose[5] -= swing_2 * 1.2
    dynamic_pose[7] += swing_2 * 1.5
    dynamic_pose[8] -= swing_2 * 1.2

    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = dynamic_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

# ============================================================
# CONFIGURAZIONE WEBCAM + YUNET
# ============================================================
yunet_model_path = cv2.samples.findFile('face_detection_yunet_2023mar.onnx')
try:
    face_detector = cv2.FaceDetectorYN.create(yunet_model_path, "", (320, 320), 0.9, 0.3, 5000)
    print("[OK] Modello YuNet caricato correttamente")
except cv2.error:
    print("[ERRORE] Impossibile caricare YuNet.")
    exit(1)
        
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)
print("[OK] Webcam attiva")

cv2.namedWindow('Webcam - Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam - Face Detection', 640, 480)
cv2.moveWindow('Webcam - Face Detection', 50, 50)

# ============================================================
# SETUP INIZIALE MUJOCO
# ============================================================
data.qpos[2] = current_height
data.qpos[7:19] = neutral_pose
mujoco.mj_forward(model, data)

print(f"\nNumero di motori: {model.nu}")
print(f"Altezza iniziale: {data.qpos[2]:.3f} m")
print("\n" + "=" * 50)
print("SISTEMA RILEVAMENTO VOLTO (YuNet):")
print("  - Robot in posa neutra")
print("  - Rilevamento volto attivo")
print("  - Rotazione proporzionale alla distanza")
print("=" * 50)

# ============================================================
# LOOP PRINCIPALE
# ============================================================
print("\nAvvio simulazione...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = [0.35, 0.0, 0.35]
    viewer.cam.azimuth = 180.0
    viewer.cam.elevation = -10.0
    
    while viewer.is_running():
        step_start = time.perf_counter()
        current_time = time.time()
        
        if current_time - last_vision_time > VISION_INTERVAL:
            last_vision_time = current_time
            ret, frame = cap.read()
            
            if ret and frame is not None:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                
                face_detector.setInputSize((w, h))
                retval, faces = face_detector.detect(frame)
                face_detected = isinstance(faces, np.ndarray) and faces.shape[0] > 0
                
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                
                if face_detected:
                    x, y, w_face, h_face = faces[0][:4].astype(int)
                    fx = x + w_face // 2
                    fy = y + h_face // 2
                    
                    dist = math.sqrt((fx - cx)**2 + (fy - cy)**2)
                    offset = fx - cx
                    
                    # ROTAZIONE PROPORZIONALE: 0.025 radianti per ogni 50 pixel di offset
                    # Il segno meno inverte la direzione per far seguire correttamente il volto
                    target_yaw = -(offset / 50.0) * 0.025
                    
                    model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]
                    model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
                    
                    cv2.line(frame, (cx, cy), (fx, fy), (0, 255, 0), 2)
                    cv2.circle(frame, (fx, fy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"Dist: {int(dist)} px | Yaw: {target_yaw:.3f}", (cx + 15, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
                else:
                    target_yaw = 0.0
                    model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
                    model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
                
                status_text = f"Volto: {'RILEVATO' if face_detected else 'ASSENTE'}"
                color = (0, 255, 0) if face_detected else (0, 0, 255)
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow('Webcam - Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

cap.release()
cv2.destroyAllWindows()
print("\nSimulazione terminata.")

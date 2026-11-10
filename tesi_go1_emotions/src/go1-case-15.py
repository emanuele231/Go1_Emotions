
# Dinamico-CL, Luci-on-Random, Closed-World

import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import sys
import termios
import tty
import select
import cv2
import math

# ============================================================
# CARICAMENTO MODELLO
# ============================================================
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/Go1_Emotions/scene.xml"
print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.iterations = 200
last_real_time = time.perf_counter()

headlight_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_left')
headlight_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_right')
if headlight_left_id == -1 or headlight_right_id == -1:
    print("[ERRORE] Fari non trovati nel modello.")
    exit(1)

mujoco.mj_forward(model, data)

# ============================================================
# MOVIMENTI
# ============================================================
animations = {
    "sad": {
        "keyframes": [
            np.array([0.0,  0.7, -1.7,  0.0,  0.7, -1.7,  0.0,  1.0, -2.2,  0.0,  1.0, -2.2]),
            np.array([0.0,  0.8, -1.9,  0.0,  0.8, -1.9,  0.0,  1.1, -2.4,  0.0,  1.1, -2.4]),
            np.array([0.0,  0.95, -2.1,  0.0,  0.95, -2.1,  0.0,  0.95, -2.1,  0.0,  0.95, -2.1]),
            np.array([0.0,  1.05, -2.2,  0.0,  1.05, -2.2,  0.0,  0.9, -2.0,  0.0,  0.9, -2.0])
        ],
        "duration": 3.0,   
    },
    "happy": {
          "keyframes": [
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.05,  0.6, -1.4,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2, -0.05,  0.6, -1.4,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
        ],
        "duration": 1.0,  
    },
    "angry": {
        "keyframes": [
            np.array([0.0,  0.6, -1.5,  0.0,  0.6, -1.5,  0.0,  0.9, -2.0,  0.0,  0.9, -2.0]),
            np.array([0.0,  0.7, -1.65,  0.0,  0.7, -1.65,  0.0,  1.0, -2.1,  0.0,  1.0, -2.1]),
            np.array([0.0,  0.85, -1.85,  0.0,  0.85, -1.85,  0.0,  1.15, -2.35,  0.0,  1.15, -2.35]),
        ],
        "duration": 2.5,
    },
    "playful": {
        "keyframes": [
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5]),
            np.array([0.25,  1.3, -2.4,  0.25,  1.3, -2.4,  0.25,  0.9, -1.5,  0.25,  0.9, -1.5]),
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5]),
            np.array([-0.25,  1.3, -2.4, -0.25,  1.3, -2.4,  -0.25,  0.9, -1.5,  -0.25,  0.9, -1.5]),
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5])
        ],
        "duration": 2.5,
    }
}

neutral_stand_pose = np.array([
    0.0,  0.6, -1.5,
    0.0,  0.6, -1.5,
    0.0,  0.9, -2.0,
    0.0,  0.9, -2.0
])

# ============================================================
# STATO E CODA POSE
# ============================================================
phase = "idle"
current_emotion = None
current_animation = None
current_pose = neutral_stand_pose.copy()
start_pose = neutral_stand_pose.copy()
transition_start_time = None
TRANSITION_DURATION = 5.0

anim_elapsed = 0.0
anim_duration = 0.0
anim_keyframes = None

last_debug_print = 0.0

last_target_yaw = 0.0
current_hip_rot = 0.0

pose_queue = list(animations.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale dei movimenti")

pose_count = 0          
MAX_POSES = 5         
simulation_finished = False  

KP = 20.0
KD = 0.05

# Variabili Face Tracking
current_yaw = 0.0
target_yaw = 0.0
walk_phase = 0.0
walk_amplitude = 0.0

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def controller(data, model):
    global current_pose, start_pose, transition_start_time, current_animation
    global phase, anim_elapsed, anim_duration, anim_keyframes
    global target_yaw, last_target_yaw, walk_phase, walk_amplitude, last_real_time, current_hip_rot

    if current_animation is None:
        return

    current_real_time = time.perf_counter()
    real_dt = current_real_time - last_real_time
    last_real_time = current_real_time

    # --- GESTIONE ANIMAZIONI ---
    if phase == "to_neutral" or phase == "to_anim":
        if transition_start_time is None:
            transition_start_time = current_real_time
            start_pose = current_pose.copy()
        
        elapsed = current_real_time - transition_start_time
        t = min(elapsed / TRANSITION_DURATION, 1.0)
        t_smooth = smoothstep(t)
        
        if phase == "to_neutral":
            target = neutral_stand_pose
        else:
            target = anim_keyframes[0]
        
        current_pose = start_pose + t_smooth * (target - start_pose)
        
        if t >= 1.0:
            transition_start_time = None
            if phase == "to_neutral":
                phase = "to_anim"
            else:
                phase = "animating"
                anim_elapsed = 0.0
    
    elif phase == "animating":
        anim_elapsed += real_dt
        
        if anim_elapsed >= anim_duration:
            anim_elapsed = anim_duration
            current_pose = anim_keyframes[-1].copy()
            phase = "idle"
        else:
            num_frames = len(anim_keyframes)
            segment_duration = anim_duration / (num_frames - 1)
            idx = int(anim_elapsed / segment_duration)
            
            if idx >= num_frames - 1:
                idx = num_frames - 2
            
            t = (anim_elapsed - idx * segment_duration) / segment_duration
            t_smooth = smoothstep(t)
            
            frame1 = anim_keyframes[idx]
            frame2 = anim_keyframes[idx + 1]
            current_pose = frame1 + t_smooth * (frame2 - frame1)

    target_yaw_clamped = max(-0.5, min(0.5, target_yaw))

    face_moving = abs(target_yaw - last_target_yaw) > 0.8
    last_target_yaw = target_yaw

    hip_error = target_yaw_clamped - current_hip_rot
    current_hip_rot += hip_error * 0.08

    if face_moving and abs(target_yaw) > 0.015:
        walk_phase += 10.0 * real_dt  
        walk_amplitude = 1.0  
    else:
        walk_amplitude += (0.0 - walk_amplitude) * 0.2  

    dynamic_pose = current_pose.copy()

    dynamic_pose[0] -= current_hip_rot
    dynamic_pose[3] -= current_hip_rot

    if walk_amplitude > 0.1:
        swing_1 = math.sin(walk_phase) * walk_amplitude
        swing_2 = math.sin(walk_phase + math.pi) * walk_amplitude

        dynamic_pose[1] += swing_1 * 1.2 
        dynamic_pose[2] -= swing_1 * 1.5  
        dynamic_pose[4] += swing_2 * 1.2  
        dynamic_pose[5] -= swing_2 * 1.5  

    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = dynamic_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)
# ============================================================
# CAMBIO POSA
# ============================================================
def change_pose():
    global phase, current_emotion, current_animation, anim_keyframes, anim_duration
    global start_pose, pose_count, simulation_finished, transition_start_time
    
    if not pose_queue:
        print("\n[TUTTE LE ANIMAZIONI SONO STATE MOSTRATE]")
        simulation_finished = True
        return
    
    current_animation = pose_queue.pop()
    current_emotion = current_animation
    anim_keyframes = animations[current_animation]["keyframes"]
    anim_duration = animations[current_animation]["duration"]
    pose_count += 1
    
    # Gestione luci in base all'emozione
    if current_emotion == "sad":
        model.geom_rgba[headlight_left_id] = [0.0, 1.0, 0.0, 1.0]
        model.geom_rgba[headlight_right_id] = [0.0, 1.0, 0.0, 1.0]
    elif current_emotion in ("happy", "playful"):
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 1.0, 1.0]
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 1.0, 1.0]
    elif current_emotion == "angry": 
        model.geom_rgba[headlight_left_id] = [0.0, 1.0, 0.0, 1.0]
        model.geom_rgba[headlight_right_id] = [0.0, 1.0, 0.0, 1.0]
    
    phase = "to_neutral"
    transition_start_time = None

    if pose_count >= MAX_POSES:
        simulation_finished = True

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

VISION_INTERVAL = 0.1
last_vision_time = 0.0

# ============================================================
# LETTURA TASTO DAL TERMINALE
# ============================================================
def get_key():
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        return sys.stdin.read(1)
    return None

# ============================================================
# INIZIALIZZAZIONE
# ============================================================
old_settings = termios.tcgetattr(sys.stdin)
try:
    tty.setcbreak(sys.stdin.fileno())
    
    change_pose()
    mujoco.mj_forward(model, data)

    print("\n" + "=" * 50)
    print("  ANIMAZIONI RANDOM + FACE TRACKING")
    print(f"  Mostra {MAX_POSES-1} animazioni casuali")
    print("  Premi [P] nel TERMINALE per la prossima animazione")
    print("=" * 50)

    # ============================================================
    # LOOP PRINCIPALE
    # ============================================================
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.35, 0.0, 0.35]
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -10.0
        
        while viewer.is_running() and not simulation_finished:
            step_start = time.perf_counter()
            current_time = time.time()
            
            # --- LETTURA TASTO ---
            key = get_key()
            if key and key.lower() == 'p':
                change_pose()
            
            # --- FACE TRACKING ---
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
                        
                        offset = fx - cx
                        if abs(offset) <= 50:
                            target_yaw = 0.0
                        else:
                            target_yaw = -(offset / 50.0) * 0.15
                        
                        cv2.line(frame, (cx, cy), (fx, cy), (0, 255, 0), 2)
                        cv2.circle(frame, (fx, cy), 4, (0, 255, 0), -1)
                        cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
                    else:
                        target_yaw = 0.0
                    
                    cv2.imshow('Webcam - Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # --- FISICA E RENDERING ---
            controller(data, model)
            if time.perf_counter() - last_debug_print > 1.0:
               print(f"[DEBUG] Emozione: {current_animation} | Fase: {phase}", end="\r", flush=True)
               last_debug_print = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # --- TIMING LOOP ---
            time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    cap.release()
    cv2.destroyAllWindows()

print("\nSimulazione terminata.")

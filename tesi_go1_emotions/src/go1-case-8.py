# Dinamico-OL, Luci-on, Closed-World

import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import sys
import termios
import tty
import select

# ============================================================
# CARICAMENTO MODELLO
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
TRANSITION_DURATION = 3.0

anim_elapsed = 0.0
anim_duration = 0.0
anim_keyframes = None

last_debug_print = 0.0

pose_queue = list(animations.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale dei movimenti")

pose_count = 0          
MAX_POSES = 5         
simulation_finished = False  

KP = 20.0
KD = 0.05

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def update_headlights(color):
    for geom_id in (headlight_left_id, headlight_right_id):
        mat_id = model.geom_matid[geom_id]
        if mat_id != -1:
            model.mat_rgba[mat_id] = color
        else:
            model.geom_rgba[geom_id] = color

def controller(data, model):
    global current_pose, start_pose, transition_start_time, current_animation
    global phase, anim_elapsed, anim_duration, anim_keyframes

    if current_animation is None:
        return

    current_time = data.time

    if phase == "to_neutral" or phase == "to_anim":
        if transition_start_time is None:
            transition_start_time = current_time
            start_pose = current_pose.copy()
        
        elapsed = current_time - transition_start_time
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
        anim_elapsed += model.opt.timestep
        
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

    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = current_pose[i]
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
    
    if current_emotion == "sad":
        update_headlights([0.0, 0.0, 0.0, 0.0])
    elif current_emotion in ("happy", "playful"):
        update_headlights([0.0, 1.0, 0.0, 1.0])
    elif current_emotion == "angry": 
        update_headlights([1.0, 0.0, 0.0, 1.0])
    
    phase = "to_neutral"
    transition_start_time = None

    if pose_count >= MAX_POSES:
        simulation_finished = True

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
    print("  POSE RANDOM GO1 (senza ripetizioni)")
    print(f"  Mostra {MAX_POSES-1} pose casuali, poi termina")
    print("  Premi [P] nel TERMINALE per la prossima posa")
    print("=" * 50)

    # ============================================================
    # LOOP PRINCIPALE
    # ============================================================
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.2]
        
        while viewer.is_running() and not simulation_finished:
            step_start = time.perf_counter()
            
            key = get_key()
            if key and key.lower() == 'p':
                change_pose()
            
            controller(data, model)
            if time.perf_counter() - last_debug_print > 1.0:
               print(f"[DEBUG] Emozione: {current_animation} | Fase: {phase}", end="\r", flush=True)
               last_debug_print = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

print("\nSimulazione terminata.")

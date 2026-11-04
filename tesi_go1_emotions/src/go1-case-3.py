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


mujoco.mj_forward(model, data)

# ============================================================
# MOVIMENTI
# ============================================================
animations = {
    "sad": {
        "keyframes": [
            np.array([0.0,  1.4, -2.9,  0.0,  1.4, -2.9,  0.0,  2.0, -2.9,  0.0,  2.0, -2.9]),
            np.array([-0.05,  1.45, -2.85,  0.0,  1.4, -2.9,  0.0,  2.0, -2.9,  0.0,  2.0, -2.9]),
            np.array([0.0,  1.4, -2.9,  0.0,  1.4, -2.9,  0.0,  2.0, -2.9,  0.0,  2.0, -2.9]),
            np.array([0.0,  1.4, -2.9,  0.05,  1.45, -2.85,  0.0,  2.0, -2.9,  0.0,  2.0, -2.9]),
            np.array([0.0,  1.4, -2.9,  0.0,  1.4, -2.9,  0.0,  2.1, -3.0,  0.0,  2.1, -3.0]),
            np.array([0.0,  1.4, -2.9,  0.0,  1.4, -2.9,  0.0,  2.2, -3.1,  0.0,  2.2, -3.1])
        ],
        "duration": 4.0,   
    },
    "happy": {
          "keyframes": [
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.05,  0.6, -1.4,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2, -0.05,  0.6, -1.4,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8])
        ],
        "duration": 0.6,  
    },
    "angry": {
        "keyframes": [
            np.array([0.0,  1.2, -2.2,  0.0,  1.2, -2.2,  0.0,  1.9, -2.7,  0.0,  1.9, -2.7]),
            np.array([0.0,  1.1, -2.0,  0.0,  1.1, -2.0,  0.0,  1.8, -2.6,  0.0,  1.8, -2.6]),
            np.array([0.0,  1.2, -2.2,  0.0,  1.2, -2.2,  0.0,  1.9, -2.7,  0.0,  1.9, -2.7]),
        ],
        "duration": 1.5,
    },
    "playful": {
        "keyframes": [
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.8, -1.3,  0.0,  0.8, -1.3]),
            np.array([0.3,  1.4, -2.3,  -0.3,  1.4, -2.3,  0.0,  0.7, -1.2,  0.0,  0.7, -1.2]),
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.8, -1.3,  0.0,  0.8, -1.3]),
            np.array([-0.3,  1.4, -2.3,  0.3,  1.4, -2.3,  0.0,  0.9, -1.4,  0.0,  0.9, -1.4]),
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
current_emotion = None
target_pose = neutral_stand_pose.copy()
current_pose = neutral_stand_pose.copy()
start_pose = neutral_stand_pose.copy()
transition_start_time = time.perf_counter()
animation_start_time = time.perf_counter()
current_animation = None
current_time = time.perf_counter()
last_debug_print = 0.0
last_printed_idx = -1
TRANSITION_DURATION = 3.0

pose_queue = list(animations.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale dei movimenti")

pose_count = 0          
MAX_POSES = 5         
simulation_finished = False  


KP = 20.0
KD = 0.0

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def controller(data, model):
    global current_pose, transition_start_time, animation_start_time, current_animation, last_printed_idx

    if current_animation is None:
        return

    current_time = data.time

    if transition_start_time is not None:
        elapsed = current_time - transition_start_time
        t = min(elapsed / TRANSITION_DURATION, 1.0)
        
        t_smooth = smoothstep(t)
        target_start = animations[current_animation]["keyframes"][0]
        current_pose = start_pose + t_smooth * (target_start - start_pose)
        
        if t >= 1.0:
            transition_start_time = None
            animation_start_time = current_time
            
    else:
        if animation_start_time is None:
            animation_start_time = current_time
            
        anim = animations[current_animation]
        keyframes = anim["keyframes"]
        duration = anim["duration"]
        num_frames = len(keyframes)
        
        elapsed = current_time - animation_start_time
        cycle_time = elapsed % duration
        
        segment_duration = duration / num_frames
        idx = int(cycle_time / segment_duration)
        
        if idx >= num_frames:
            idx = num_frames - 1
            
            
        t = (cycle_time - idx * segment_duration) / segment_duration
        t_smooth = smoothstep(t)
        
        frame1 = keyframes[idx]
        frame2 = keyframes[(idx + 1) % num_frames]
        
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
    global current_emotion, current_animation, start_pose, pose_count, simulation_finished, transition_start_time, animation_start_time
    
    if not pose_queue:
        print("\n[TUTTE LE ANIMAZIONI SONO STATE MOSTRATE]")
        simulation_finished = True
        return
    
    start_pose = current_pose.copy()
    
    current_animation = pose_queue.pop()
    current_emotion = current_animation
    pose_count += 1
    
    animation_start_time = None
    transition_start_time = data.time

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
            
            # --- LETTURA TASTO ---
            key = get_key()
            if key and key.lower() == 'p':
                change_pose()
            
            # --- FISICA E RENDERING ---
            controller(data, model)
            if time.perf_counter() - last_debug_print > 1.0:
               print(f"[DEBUG] Emozione: {current_animation}", end="\r", flush=True)
               last_debug_print = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # --- TIMING LOOP ---
            time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

print("\nSimulazione terminata.")

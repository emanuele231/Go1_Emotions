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
# POSE
# ============================================================
poses = {
    "sad": np.array([
        0.0,  1.4, -2.9,
        0.0,  1.4, -2.9,
        0.0,  2.0, -2.9,
        0.0,  2.0, -2.9
    ]),
    "happy": np.array([
        0.0,  0.8, -1.2,
        0.0,  0.8, -1.2,
        0.0,  1.0, -1.8,
        0.0,  1.0, -1.8
    ]),
    "angry": np.array([
        0.0,  1.2, -2.2,
        0.0,  1.2, -2.2,
        0.0,  1.9, -2.7,
        0.0,  1.9, -2.7
    ]),
    "playful": np.array([
        0.0,  1.3, -2.4,
        0.0,  1.3, -2.4,
        0.0,  0.8, -1.3,
        0.0,  0.8, -1.3
    ])
}

neutral_stand_pose = np.array([
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5,
    0.0,  0.9, -1.5
])

# ============================================================
# STATO E CODA POSE
# ============================================================
current_emotion = None
target_pose = neutral_stand_pose.copy()
current_pose = neutral_stand_pose.copy()
start_pose = neutral_stand_pose.copy()
transition_start_time = time.perf_counter()
TRANSITION_DURATION = 3.0

pose_queue = list(poses.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale delle pose")

pose_count = 0          
MAX_POSES = 5         
simulation_finished = False  


KP = 20.0
KD = 0.0

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def controller(data, model):
    global current_pose, transition_start_time
    
    if transition_start_time is not None:
        elapsed = time.perf_counter() - transition_start_time
        t = min(elapsed / TRANSITION_DURATION, 1.0)
        t_smooth = smoothstep(t)
        current_pose = start_pose + t_smooth * (target_pose - start_pose)
        if t >= 1.0:
            transition_start_time = None
    else:
        current_pose = target_pose.copy()
    
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = current_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

# ============================================================
# CAMBIO POSA
# ============================================================
def change_pose():
    global current_emotion, target_pose, start_pose, pose_count, simulation_finished, transition_start_time
    
    if not pose_queue:
        print("\n[TUTTE LE POSE SONO STATE MOSTRATE]")
        simulation_finished = True
        return
    
    start_pose = current_pose.copy()
    
    current_emotion = pose_queue.pop()
    target_pose = poses[current_emotion].copy()
    pose_count += 1
    model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
    model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
    
    transition_start_time = time.perf_counter()
    print(f"\r[{time.strftime('%H:%M:%S')}] >>> Posa {pose_count}/{MAX_POSES-1}    ", end="", flush=True)

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
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # --- TIMING LOOP ---
            time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

print("\nSimulazione terminata.")

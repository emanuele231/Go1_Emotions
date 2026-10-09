import mujoco
import mujoco.viewer
import numpy as np
import time

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

# === POSE DEFINITE ===
sad_pose = np.array([
     0.0,  0.4, -1.8,
     0.0,  0.4, -1.8,
     0.0,  0.3, -1.7,
     0.0,  0.3, -1.7
])

ready_pose = np.array([
    0.0, 0.5, -1.4,
    0.0, 0.5, -1.4,
    0.0, 0.9, -1.6, 
    0.0, 0.9, -1.6
])

# === STATO CORRENTE ===
current_state = "sad"

# === INTERPOLAZIONE TRANSIZIONE ===
transition_duration = 1.2
transition_start_time = None
start_pose = sad_pose.copy()
target_pose = sad_pose.copy()
interpolated_pose = sad_pose.copy()

# === CONTROLLO PD ===
KP = 80.0
KD = 0.0

def smoothstep(t):
    """Curva ease-in-ease-out per movimenti naturali"""
    return t * t * (3 - 2 * t)

def apply_state(state_name):
    global current_state, start_pose, target_pose, transition_start_time
    
    # Salva la posa corrente come punto di partenza
    start_pose = interpolated_pose.copy()
    
    if state_name == "sad":
        target_pose = sad_pose.copy()
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: SAD")
        
    elif state_name == "ready":
        target_pose = ready_pose.copy()
        model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]
        model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione verso: READY")
    
    current_state = state_name
    transition_start_time = time.time()

def controller(data, model):
    """Interpola la posa e mantiene con controllo PD"""
    global interpolated_pose, transition_start_time
    
    # Calcola l'interpolazione se c'è una transizione in corso
    if transition_start_time is not None:
        elapsed = time.time() - transition_start_time
        t = min(elapsed / transition_duration, 1.0)
        
        # Applica curva smooth
        t_smooth = smoothstep(t)
        
        # Interpola linearmente tra start e target
        interpolated_pose = start_pose + t_smooth * (target_pose - start_pose)
        
        # Se la transizione è completa, ferma il timer
        if t >= 1.0:
            transition_start_time = None
    
    # Controllo PD sulla posa interpolata
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = interpolated_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

# Applicare stato iniziale
apply_state("sad")

# === SETUP INIZIALE ===
data.qpos[2] = 0.30
data.qpos[7:19] = interpolated_pose
mujoco.mj_forward(model, data)

print(f"Numero di motori: {model.nu}")
print(f"Altezza base: {data.qpos[2]:.3f} m")
print("\n" + "=" * 50)
print("CONTROLLI TASTIERA:")
print("  [A] -> Stato SAD    (fari spenti, posa raccolta)")
print("  [B] -> Stato READY  (fari accesi, posa distesa)")
print("  [ESC] -> Esci")
print("=" * 50)

def key_callback(keycode):
    """Gestisce i tasti premuti dall'utente"""
    if keycode == 65:  # A
        apply_state("sad")
    elif keycode == 66:  # B
        apply_state("ready")

print("\nAvvio simulazione...")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    while viewer.is_running():
        step_start = time.perf_counter()
        
        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if int(data.time / model.opt.timestep) % 1000 == 0:
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m | Stato: {current_state.upper()}")
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\nSimulazione terminata.")

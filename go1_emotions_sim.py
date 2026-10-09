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
     0.0,  0.4, -1.8,   # Front Right (più piegato, testa bassa)
     0.0,  0.4, -1.8,   # Front Left
     0.0,  0.3, -1.7,   # Rear Right
     0.0,  0.3, -1.7    # Rear Left
])

ready_pose = np.array([
    0.0, 0.4, -1.0,     # Front (più distese, testa alta)
    0.0, 0.4, -1.0,
    0.0, 0.3, -1.0, 
    0.0, 0.3, -1.0
])

# === STATO CORRENTE ===
current_state = "sad"
current_pose = sad_pose.copy()

# Funzione per applicare lo stato (posa + fari)
def apply_state(state_name):
    global current_state, current_pose
    
    if state_name == "sad":
        current_pose = sad_pose.copy()
        # Fari SPENTI (trasparenti)
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Stato: SAD (Fari spenti)")
        
    elif state_name == "ready":
        current_pose = ready_pose.copy()
        # Fari ACCESI (bianchi)
        model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]
        model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
        print(f"\n[{time.strftime('%H:%M:%S')}] >>> Stato: READY (Fari accesi)")
    
    current_state = state_name

# Applicare stato iniziale
apply_state("sad")

# === SETUP INIZIALE ===
data.qpos[2] = 0.30
data.qpos[7:19] = current_pose
mujoco.mj_forward(model, data)

print(f"Numero di motori: {model.nu}")
print(f"Altezza base: {data.qpos[2]:.3f} m")
print("\n" + "=" * 50)
print("CONTROLLI TASTIERA:")
print("  [1] -> Stato SAD    (fari spenti, posa raccolta)")
print("  [2] -> Stato READY  (fari accesi, posa distesa)")
print("  [ESC] -> Esci")
print("=" * 50)

# === CONTROLLO PD ===
KP = 100.0
KD = 0.0

def controller(data, model):
    """Mantiene la posa corrente con controllo PD"""
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = current_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

# === CALLBACK TASTIERA ===
def key_callback(keycode):
    """Gestisce i tasti premuti dall'utente"""
    # Codici GLFW: '1' = 49, '2' = 50
    if keycode == 49:  # Tasto 1
        apply_state("sad")
    elif keycode == 50:  # Tasto 2
        apply_state("ready")

# === AVVIO SIMULAZIONE ===
print("\nAvvio simulazione...")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    while viewer.is_running():
        step_start = time.perf_counter()
        
        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Diagnostica ogni secondo
        if int(data.time / model.opt.timestep) % 1000 == 0:
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m | Stato: {current_state.upper()}")
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\nSimulazione terminata.")

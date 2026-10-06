import mujoco
import mujoco.viewer
import numpy as np
import time

# Carica la scena di Menagerie (ha già pavimento, luci, telecamera)
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/scene.xml"

print(f"Caricamento modello: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# === POSA NEUTRA IN PIEDI ===
# Altezza della base (z)
data.qpos[2] = 0.32

# Posizioni dei 12 joint per stare in piedi (hip, thigh, calf per ogni zampa)
# Ordine: FR (Front Right), FL (Front Left), RR (Rear Right), RL (Rear Left)
neutral_pose = np.array([
     0.1,  0.8, -1.5,   # Front Right: hip, thigh, calf
    -0.1,  0.8, -1.5,   # Front Left
     0.1,  0.8, -1.5,   # Rear Right
    -0.1,  0.8, -1.5    # Rear Left
])

# Applica la posa neutra
data.qpos[7:19] = neutral_pose

# Inizializza la fisica
mujoco.mj_forward(model, data)

print(f"Numero di motori: {model.nu}")
print(f"Altezza base: {data.qpos[2]:.3f} m")
print(f"Posa joint: {data.qpos[7:19]}")

# === CONTROLLO PD PER MANTENERE LA POSA ===
KP = 200.0  # Rigidità
KD = 10.0   # Smorzamento

def controller(data, model):
    """Mantiene la posa neutra con controllo PD"""
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = neutral_pose[i]
        
        # Calcolo coppia: PD control
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

print("\nAvvio simulazione. Il robot dovrebbe stare fermo in piedi.")
print("Premi ESC per uscire.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Imposta la telecamera
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    while viewer.is_running():
        step_start = time.perf_counter()
        
        # Applica controllo
        controller(data, model)
        
        # Avanza simulazione
        mujoco.mj_step(model, data)
        
        # Aggiorna visualizzazione
        viewer.sync()
        
        # Diagnostica ogni secondo
        if int(data.time / model.opt.timestep) % 1000 == 0:
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m | contatti={data.ncon}")
        
        # Mantieni realtime
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\nSimulazione terminata.")

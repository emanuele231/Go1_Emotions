import mujoco
import mujoco.viewer
import numpy as np
import time

model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/scene.xml"

print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

model.opt.iterations = 200

mujoco.mj_forward(model, data)

# === POSA NEUTRA ===
neutral_pose = np.array([
     0.0,  0.9, -1.8,   # Front Right (hip, thigh, calf)
     0.0,  0.9, -1.8,   # Front Left
     0.0,  0.9, -1.8,   # Rear Right
     0.0,  0.9, -1.8    # Rear Left
])

# Setup iniziale
data.qpos[2] = 0.26
data.qpos[7:19] = neutral_pose
mujoco.mj_forward(model, data)

print(f"Numero di motori: {model.nu}")
print(f"Altezza base: {data.qpos[2]:.3f} m")

# Parametri di controllo PD
KP = 100.0
KD = 0.0

def controller(data, model):
    """Mantiene la posa neutra con controllo PD"""
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = neutral_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

print("\nAvvio simulazione. Il robot starà fermo in posizione neutra.")
print("Premi ESC per uscire.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    while viewer.is_running():
        step_start = time.perf_counter()
        
        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Diagnostica ogni secondo
        if int(data.time / model.opt.timestep) % 1000 == 0:
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m")
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

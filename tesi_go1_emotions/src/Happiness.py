import mujoco
import mujoco.viewer
import numpy as np
import time

model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/scene.xml"

print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

model.opt.iterations = 200

#Setting di luci e luci colorate
headlight_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_left')
headlight_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'headlight_right')

if headlight_left_id != -1 and headlight_right_id != -1:
    # CORREZIONE: Si usa model.geom_rgba (proprietà statica), non data.geom_rgba
    model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]
    model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
    print("[OK] Fari bianchi ACCESI!")
else:
    print("[ERRORE] Fari non trovati nel modello.")

mujoco.mj_forward(model, data)

# === POSA FELICITÀ ===
# Il robot è più alto, con zampe più raccolte (energico, pronto a saltare)
happiness_pose = np.array([
     0.0,  0.9, -1.5,   # Front Right (thigh più piegato)
     0.0,  0.9, -1.5,   # Front Left
     0.0,  1.4, -2.0,   # Rear Right
     0.0,  1.4, -2.0    # Rear Left
])

# Setup iniziale - altezza leggermente più alta per posa felice
data.qpos[2] = 0.30
data.qpos[7:19] = happiness_pose
mujoco.mj_forward(model, data)

print(f"Numero di motori: {model.nu}")
print(f"Altezza base: {data.qpos[2]:.3f} m")
print(f"Emozione: FELICITÀ")

# Parametri di controllo PD (stessi del codice funzionante)
KP = 100.0
KD = 0.0

def controller(data, model):
    """Mantiene la posa felice con controllo PD"""
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = happiness_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

print("\nAvvio simulazione. Il robot esprimerà FELICITÀ.")
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
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m | Emozione: FELICITÀ")
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

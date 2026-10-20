import mujoco
import mujoco.viewer
import numpy as np
import time

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
# POSA SAD (TRISTEZZA)
# ============================================================
sad_pose = np.array([
    0.0,  1.4, -2.9,   
    0.0,  1.4, -2.9,  
    0.0,  2.0, -2.9,   
    0.0,  2.0, -2.9  
])

happy_pose = np.array([
    0.0,  0.8, -1.2,   
    0.0,  0.8, -1.2,   
    0.0,  1.0, -1.8,   
    0.0,  1.0, -1.8   
])

angry_pose = np.array([
    0.0,  1.2, -2.2,   
    0.0,  1.2, -2.2,   
    0.0,  1.9, -2.7,   
    0.0,  1.9, -2.7   
])

playful_pose = np.array([
    0.0,  1.3, -2.4,   
    0.0,  1.3, -2.4,   
    0.0,  0.8, -1.3,   
    0.0,  0.8, -1.3    
])

confused_pose = np.array([
     0.4,  1.1, -2.0,   
    -0.1,  1.1, -2.0,  
     0.0,  1.3, -1.9,   
     0.0,  1.3, -1.9    
])

sad_height = 0.32  

# ============================================================
# CONTROLLO
# ============================================================
KP = 20.0
KD = 0.0

def controller(data, model):
    # PD controller per i 12 giunti
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = confused_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

# ============================================================
# INIZIALIZZAZIONE
# ============================================================
data.qpos[2] = 0.32  # altezza iniziale (neutra)
data.qpos[7:19] = sad_pose
mujoco.mj_forward(model, data)

model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]

print("\n" + "=" * 50)
print("  POSA SAD (TRISTEZZA)")
print("=" * 50)

# ============================================================
# LOOP PRINCIPALE
# ============================================================
print("\nAvvio simulazione...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    while viewer.is_running():
        step_start = time.perf_counter()
        
        

        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if int(data.time / model.opt.timestep) % 1000 == 0:
            print(f"t={data.time:.2f}s | z={data.qpos[2]:.3f}m")

        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\nSimulazione terminata.")

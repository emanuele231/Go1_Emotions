import mujoco
import mujoco.viewer
import numpy as np
import time
import os

model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/scene.xml"

print("\n" + "="*60)
print("DIAGNOSTICA SCENA MUJOCO")
print("="*60)
print(f"Percorso file: {model_path}")

# Verifica che il file esista
if not os.path.exists(model_path):
    print("[ERRORE] Il file non esiste!")
    exit(1)
print(f"[OK] File esistente, dimensione: {os.path.getsize(model_path)} bytes")

# Carica il modello
try:
    model = mujoco.MjModel.from_xml_path(model_path)
    print("[OK] Modello caricato correttamente")
except Exception as e:
    print(f"[ERRORE] Errore nel caricamento: {e}")
    exit(1)

data = mujoco.MjData(model)

# === VERIFICA 1: Pavimento ===
print("\n" + "="*60)
print("VERIFICA 1: PAVIMENTO")
print("="*60)

floor_found = False
for i in range(model.ngeom):
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        pos = model.geom_pos[i]
        contype = model.geom_contype[i]
        conaffinity = model.geom_conaffinity[i]
        
        print(f"[OK] Pavimento trovato: {geom_name}")
        print(f"   Posizione: {pos}")
        print(f"   contype: {contype}")
        print(f"   conaffinity: {conaffinity}")
        
        if contype > 0 and conaffinity > 0:
            print(f"   [OK] Collisioni ATTIVE")
        else:
            print(f"   [ERRORE] Collisioni DISATTIVATE (contype={contype}, conaffinity={conaffinity})")
        
        floor_found = True
        break

if not floor_found:
    print("[ERRORE] NESSUN PAVIMENTO TROVATO!")
    print(f"   Geom totali: {model.ngeom}")
    for i in range(min(10, model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"   Geom {i}: {name}, type={model.geom_type[i]}")

# === VERIFICA 2: Posizione iniziale del robot ===
print("\n" + "="*60)
print("VERIFICA 2: POSIZIONE ROBOT")
print("="*60)

mujoco.mj_forward(model, data)

robot_height = data.qpos[2]
print(f"Altezza robot (z): {robot_height:.3f} m")

if robot_height > 0.1:
    print("[OK] Robot posizionato sopra il pavimento")
else:
    print("[ERRORE] Robot troppo basso o sotto il pavimento!")

# === VERIFICA 3: Contatti iniziali ===
print("\n" + "="*60)
print("VERIFICA 3: CONTATTI INIZIALI")
print("="*60)

print(f"Numero contatti dopo mj_forward: {data.ncon}")

if data.ncon > 0:
    print("[OK] Ci sono contatti attivi")
    for i in range(min(5, data.ncon)):
        contact = data.contact[i]
        print(f"   Contatto {i}: geom1={contact.geom1}, geom2={contact.geom2}")
else:
    print("[ATTENZIONE] Nessun contatto iniziale (il robot potrebbe essere troppo in alto)")

# === VERIFICA 4: Gravità ===
print("\n" + "="*60)
print("VERIFICA 4: GRAVITA")
print("="*60)

gravity = model.opt.gravity
print(f"Gravita: {gravity}")
if gravity[2] < 0:
    print("[OK] Gravita diretta verso il basso")
else:
    print("[ERRORE] Gravita anomala!")

# === RIEPILOGO ===
print("\n" + "="*60)
print("RIEPILOGO")
print("="*60)

issues = []
if not floor_found:
    issues.append("Manca il pavimento")
if robot_height <= 0.1:
    issues.append("Robot troppo basso")
if data.ncon == 0 and robot_height < 0.5:
    issues.append("Nessun contatto nonostante altezza bassa")

if issues:
    print("[ERRORE] Problemi trovati:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nLa simulazione potrebbe non funzionare correttamente.")
else:
    print("[OK] Tutto sembra OK, avvio la simulazione...")

# === SIMULAZIONE ===
print("\n" + "="*60)
print("AVVIO SIMULAZIONE")
print("="*60)

KP = 200.0
KD = 10.0
initial_pose = data.qpos[7:7+model.nu].copy()

def controller(data, model):
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = initial_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    step_count = 0
    while viewer.is_running():
        step_start = time.perf_counter()
        controller(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step {step_count}: t={data.time:.2f}s | z={data.qpos[2]:.3f}m | contatti={data.ncon}")
            
            if data.qpos[2] < -1.0:
                print(f"[ERRORE] Robot caduto nel vuoto! z={data.qpos[2]:.3f}")
                break
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("\nSimulazione terminata.")

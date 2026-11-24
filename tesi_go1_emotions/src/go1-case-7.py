# Dinamico-OL, Luci-off, Closed-World + Questionari (Fase Verde + Luci Spente in Transizione)

import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import sys
import termios
import tty
import select
import warnings
import threading
import tkinter as tk
import glfw
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
last_status = ""

# ============================================================
# CARICAMENTO MODELLO
# ============================================================
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/Go1_Emotions/scene.xml"
print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.iterations = 200

# Salviamo i colori originali delle luci della scena
luci_diffuse_originali = model.light_diffuse.copy()
luci_ambient_originali = model.light_ambient.copy()

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
        "repeats": 1
    },
    "happy": {
        "keyframes": [
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.05,  0.6, -1.4,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2, -0.05,  0.6, -1.4,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.05,  0.6, -1.4,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2, -0.05,  0.6, -1.4,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
            np.array([0.0,  0.8, -1.2,  0.0,  0.8, -1.2,  0.0,  1.0, -1.8,  0.0,  1.0, -1.8]),
        ],
        "duration": 2.0,
        "repeats": 1
    },
    "angry": {
        "keyframes": [
            np.array([0.0,  0.6, -1.5,  0.0,  0.6, -1.5,  0.0,  0.9, -2.0,  0.0,  0.9, -2.0]),
            np.array([0.0,  0.7, -1.65,  0.0,  0.7, -1.65,  0.0,  1.0, -2.1,  0.0,  1.0, -2.1]),
            np.array([0.0,  0.85, -1.85,  0.0,  0.85, -1.85,  0.0,  1.15, -2.35,  0.0,  1.15, -2.35]),
        ],
        "duration": 2.5,
        "repeats": 1
    },
    "playful": {
        "keyframes": [
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5]),
            np.array([0.25,  1.3, -2.4,  0.25,  1.3, -2.4,  0.25,  0.9, -1.5,  0.25,  0.9, -1.5]),
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5]),
            np.array([-0.25,  1.3, -2.4, -0.25,  1.3, -2.4,  -0.25,  0.9, -1.5,  -0.25,  0.9, -1.5]),
            np.array([0.25,  1.3, -2.4,  0.25,  1.3, -2.4,  0.25,  0.9, -1.5,  0.25,  0.9, -1.5]),
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5]),
            np.array([-0.25,  1.3, -2.4, -0.25,  1.3, -2.4,  -0.25,  0.9, -1.5,  -0.25,  0.9, -1.5]),
            np.array([0.0,  1.3, -2.4,  0.0,  1.3, -2.4,  0.0,  0.9, -1.5,  0.0,  0.9, -1.5])
        ],
        "duration": 3.0,
        "repeats": 1
    }
}

neutral_stand_pose = np.array([
    0.0,  0.6, -1.5,
    0.0,  0.6, -1.5,
    0.0,  0.9, -2.0,
    0.0,  0.9, -2.0
])

QUESTIONNAIRE_URLS = {
    "happy": "https://docs.google.com/forms/d/e/1FAIpQLSfNPdulqmjE7rjiiQnxNJQOzNdZ_B7AkLCZqnoPJKU2tIbq8A/viewform?usp=header",
    "angry": "https://docs.google.com/forms/d/e/1FAIpQLScQIln6OaIhfvWWWO0w8odzr-CZr9fuqa08U_bUVPZO4MiQRQ/viewform?usp=header",
    "playful": "https://docs.google.com/forms/d/e/1FAIpQLSd-4e3fj21Lms8gK9EAEMQBKGndTOmbrzbiyqCvCuqSQ-IewA/viewform?usp=header",
    "sad": "https://docs.google.com/forms/d/e/1FAIpQLSeRKHovxOjWallfwPsgWeAeZUAXAP0pvFf71Xk6ZiIdvEC_cA/viewform?usp=header"
}

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
anim_repeats = 1
total_anim_duration = 0.0

last_debug_print = 0.0

pose_queue = list(animations.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale dei movimenti")

pose_count = 0
MAX_POSES = 5
simulation_finished = False

KP = 20.0
KD = 0.05

# Variabili timer e questionario
pose_ready_time = None
POSE_DISPLAY_DURATION = 10.0
is_questionnaire_open = False
pause_start_time = 0.0
questionnaire_launched = False
user_submitted = False  

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ============================================================
# QUESTIONARIO
# ============================================================
def apri_questionario(emotion):
    global is_questionnaire_open, pause_start_time, pose_ready_time, user_submitted
    
    warnings.filterwarnings("ignore")
    if emotion not in QUESTIONNAIRE_URLS:
        return

    FORM_URL = QUESTIONNAIRE_URLS[emotion]
    print(f"\n[INFO] Apertura del questionario per '{emotion}'...")

    is_questionnaire_open = True
    pause_start_time = time.perf_counter()

    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(FORM_URL)
        
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        width = 700
        height = 800
        x = (screen_width // 2) + 100
        y = (screen_height - height) // 2
        
        driver.set_window_rect(x, y, width, height)
        
        WebDriverWait(driver, 300).until(EC.url_contains('formResponse'))
        
        user_submitted = True  # <-- SEGNA CHE L'UTENTE HA RISPOSTO
        print(f"[INFO] Questionario '{emotion}' inviato! Chiusura automatica tra 3 secondi...")
        time.sleep(3)
    except Exception as e:
        print(f"\n[ERRORE] Si è verificato un problema: {e}")
    finally:
        driver.quit()
        is_questionnaire_open = False
        if pose_ready_time is not None:
            pose_ready_time += (time.perf_counter() - pause_start_time)

# ============================================================
# MESSAGGIO SCENA 3D
# ============================================================
def disegna_testo(viewer, testo, x, y, z, colore=(1, 1, 1, 1)):
    scn = viewer.user_scn
    if scn.ngeom < scn.maxgeom:
        geom = scn.geoms[scn.ngeom]
        geom.type = mujoco.mjtGeom.mjGEOM_LABEL
        geom.pos[:] = [x, y, z]
        geom.label = testo
        geom.rgba = [*colore]
        scn.ngeom += 1

# ============================================================
# CONTROLLER
# ============================================================
def controller(data, model):
    global current_pose, start_pose, transition_start_time, current_animation
    global phase, anim_elapsed, anim_duration, anim_keyframes
    global anim_repeats, total_anim_duration, pose_ready_time, questionnaire_launched

    if current_animation is None:
        return
    
    current_real_time = time.perf_counter()

    # --- CALCOLO E STAMPA DEL TIMER DINAMICO ---
    if phase in ("to_neutral", "to_anim"):
        elapsed = current_real_time - transition_start_time if transition_start_time is not None else 0.0
        status = f"{RED}UN ATTIMO! ({min(elapsed, TRANSITION_DURATION):.1f}s){RESET}"
    elif phase in ("animating", "idle"):
        elapsed = current_real_time - pose_ready_time if pose_ready_time is not None else 0.0
        status = f"{GREEN}PRONTO! ({elapsed:.1f}s){RESET}"
    else:
        status = f"{RED}STATO SCONOSCIUTO{RESET}"

    print(f"\r{status}\033[K", end="", flush=True)

    # --- FASE 1: Transizione verso neutral o primo keyframe ---
    if phase == "to_neutral" or phase == "to_anim":
        if transition_start_time is None:
            transition_start_time = current_real_time
            start_pose = current_pose.copy()
        
        elapsed = current_real_time - transition_start_time
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
    
    # --- FASE 2: Animazione con ripetizioni ---
    elif phase == "animating":
        anim_elapsed += model.opt.timestep
        cycle_elapsed = anim_elapsed % anim_duration
        
        if anim_elapsed >= total_anim_duration:
            cycle_elapsed = anim_duration
            anim_elapsed = total_anim_duration
            current_pose = anim_keyframes[-1].copy()
            phase = "idle"
            
            if pose_ready_time is None:
                pose_ready_time = time.perf_counter()
                # Lancia il questionario SOLO quando entra nella fase verde (idle)
                if not questionnaire_launched and current_emotion in QUESTIONNAIRE_URLS:
                    questionnaire_launched = True
                    threading.Thread(target=apri_questionario, args=(current_emotion,), daemon=True).start()
        else:
            num_frames = len(anim_keyframes)
            segment_duration = anim_duration / (num_frames - 1)
            idx = int(cycle_elapsed / segment_duration)
            
            if idx >= num_frames - 1:
                idx = num_frames - 2
            
            t = (cycle_elapsed - idx * segment_duration) / segment_duration
            t_smooth = smoothstep(t)
            
            frame1 = anim_keyframes[idx]
            frame2 = anim_keyframes[idx + 1]
            current_pose = frame1 + t_smooth * (frame2 - frame1)

    # --- CONTROLLO PD ---
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
    global anim_repeats, total_anim_duration, pose_ready_time, questionnaire_launched, user_submitted
    
    # Resetta i flag per la nuova emozione
    questionnaire_launched = False
    user_submitted = False
    pose_ready_time = None
    
    if not pose_queue:
        print("\n[TUTTE LE ANIMAZIONI SONO STATE MOSTRATE]")
        return
    
    start_pose = current_pose.copy()
    
    current_animation = pose_queue.pop()
    current_emotion = current_animation
    anim_keyframes = animations[current_animation]["keyframes"]
    anim_duration = animations[current_animation]["duration"]
    anim_repeats = animations[current_animation].get("repeats", 1)
    total_anim_duration = anim_duration * anim_repeats
    pose_count += 1
    
    model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
    model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
    
    phase = "to_neutral"
    transition_start_time = None

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
    
    data.qpos[7:19] = neutral_stand_pose
    mujoco.mj_forward(model, data)

    change_pose()


    # ============================================================
    # LOOP PRINCIPALE
    # ============================================================
    glfw.init()
    glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = 1.4
        viewer.cam.lookat[:] = [0.35, 0.0, 0.35]
        viewer.cam.azimuth = 175.0
        viewer.cam.elevation = -45.0
        
        while viewer.is_running() and not simulation_finished:
            step_start = time.perf_counter()
            
            # --- TIMER AUTOMATICO (Salta se l'utente ha inviato) ---
            if pose_ready_time is not None:
                tempo_trascorso = time.perf_counter() - pose_ready_time
                if tempo_trascorso >= POSE_DISPLAY_DURATION or user_submitted:
                    if not pose_queue:
                        if not is_questionnaire_open:
                            simulation_finished = True
                    else:
                        if not is_questionnaire_open:
                            change_pose()
            
            # --- MESSAGGIO NELLA SCENA 3D ---
            viewer.user_scn.ngeom = 0
            if phase in ("to_neutral", "to_anim"):
                disegna_testo(viewer, "Attendi, il robot si sta posizionando!", 1.9, -0.1, 0.6, (1.0, 0.2, 0.2, 1.0))

            # --- GESTIONE LUCI SCENA (Spente in transizione) ---
            if phase in ("to_neutral", "to_anim"):
                for i in range(model.nlight):
                    model.light_diffuse[i] = [0.0, 0.0, 0.0]
                    model.light_ambient[i] = [0.0, 0.0, 0.0]
            else:
                model.light_diffuse[:] = luci_diffuse_originali
                model.light_ambient[:] = luci_ambient_originali
            
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

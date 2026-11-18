
#Statico, Luci-off, Open-World

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
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import glfw
import tkinter as tk

import threading

# Serviranno nell'inserimento dei form
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

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

QUESTIONNAIRE_URLS = {
    "happy": "https://docs.google.com/forms/d/e/1FAIpQLSeAGuJB2G9iMkfB1IYPAwuxJJSsI1wf9eVMwYRE_miJongf1g/viewform?usp=header",
    "angry": "https://docs.google.com/forms/d/e/1FAIpQLSfQQDpmCAYghL0C7Kd1QJQZ9kZs9zoeCymXeNZWyJEOqlP6fA/viewform?usp=header",
    "playful": "https://docs.google.com/forms/d/e/1FAIpQLSeR-sQorgZK_tML3KMPpmE0YuC7wjOvdVyHfzeJ_DLklBqwAg/viewform?usp=header",
    "sad": "https://docs.google.com/forms/d/e/1FAIpQLScWFMbJoIE_h8oXlIRuUG6htu1T2fW6un2MhKi_ZjMJRx2QUw/viewform?usp=header"
}

# ============================================================
# STATO E CODA POSE
# ============================================================
current_emotion = None
target_pose = neutral_stand_pose.copy()
current_pose = neutral_stand_pose.copy()
start_pose = neutral_stand_pose.copy()
transition_start_time = time.perf_counter()
TRANSITION_DURATION = 4.0

pose_queue = list(poses.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale delle pose")

pose_count = 0          
MAX_POSES = 5         
simulation_finished = False 
questionnaire_opened = False
is_questionnaire_open = False
pause_start_time = 0.0  

pose_ready_time = None
POSE_DISPLAY_DURATION = 10.0

emotion_to_question = None  


KP = 20.0
KD = 0.0

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def controller(data, model):
    global current_pose, transition_start_time, pose_ready_time
    
    if transition_start_time is not None:
        elapsed = time.perf_counter() - transition_start_time
        t = min(elapsed / TRANSITION_DURATION, 1.0)
        t_smooth = smoothstep(t)
        current_pose = start_pose + t_smooth * (target_pose - start_pose)
        
        if t >= 1.0:
            transition_start_time = None
            pose_ready_time = time.perf_counter()
            
            if is_questionnaire_open:
                pause_start_time = time.perf_counter()
                print(f"\r{GREEN}PRONTO!{RESET}    ", end="", flush=True)
    else:
        current_pose = target_pose.copy()
    
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = current_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

# ============================================================
# DOMANDA 
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
# QUESTIONARIO
# ============================================================
def apri_questionario(emotion):
    global is_questionnaire_open, pause_start_time, pose_ready_time
    
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
        
        width = 1500
        height = 1400
        x = (screen_width // 2) + 1000
        y = (screen_height - height) // 2
        
        driver.set_window_rect(x, y, width, height)
        
        WebDriverWait(driver, 300).until(EC.url_contains('formResponse'))
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
# CAMBIO POSA
# ============================================================
def change_pose():
    global current_emotion, target_pose, start_pose, pose_count, simulation_finished, transition_start_time, pose_ready_time, emotion_to_question
    
    # Salviamo l'emozione appena conclusa prima di sovrascriverla
    emotion_to_question = current_emotion
    
    pose_ready_time = None
    
    if not pose_queue:
        print("\n[TUTTE LE POSE SONO STATE MOSTRATE]")
        return
    
    start_pose = current_pose.copy()

    # --- LOGICA FARI IN BASE ALL'EMOZIONE ---
    current_emotion = pose_queue.pop()
    target_pose = poses[current_emotion].copy()
    pose_count += 1
    model.geom_rgba[headlight_left_id] = [0.0, 0.0, 0.0, 0.0]
    model.geom_rgba[headlight_right_id] = [0.0, 0.0, 0.0, 0.0]
    
    transition_start_time = time.perf_counter()
    print(f"\r{RED}UN ATTIMO!{RESET}    ", end="", flush=True)
    
    if emotion_to_question in QUESTIONNAIRE_URLS:
        threading.Thread(target=apri_questionario, args=(emotion_to_question,), daemon=True).start()

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
    
    data.qpos[7:19] = neutral_stand_pose
    mujoco.mj_forward(model, data)

    change_pose()

    # ============================================================
    # LOOP PRINCIPALE
    # ============================================================
    glfw.init()
    glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.35, 0.0, 0.35]
        viewer.cam.azimuth = 175.0
        viewer.cam.elevation = -10.0
        
        while viewer.is_running() and not simulation_finished:
            step_start = time.perf_counter()
            
            # --- TIMER AUTOMATICO ---
            if pose_ready_time is not None and not is_questionnaire_open:
                if time.perf_counter() - pose_ready_time >= POSE_DISPLAY_DURATION:
                    if not pose_queue:
                        apri_questionario(current_emotion)
                        simulation_finished = True
                    else:
                        change_pose()
            
            # Disegna il messaggio sopra il robot (asse Z positivo)
            viewer.user_scn.ngeom = 0
            if transition_start_time is not None:
                disegna_testo(viewer, "Attendi, il robot si sta posizionando!", 0.0, -0.4, 0.2, (1.0, 0.2, 0.2, 1.0))
            else:
                disegna_testo(viewer, "Secondo te cosa sta esprimendo il robot?", -0.4, -1.0, 0.8, (1.0, 1.0, 1.0, 1.0))
            
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

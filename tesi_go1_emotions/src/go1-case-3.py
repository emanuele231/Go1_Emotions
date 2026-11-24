
#Statico, Luci-on-Random, Closed-World

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
# POSE STATICHE
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
    "happy": "https://docs.google.com/forms/d/e/1FAIpQLScMx03jctOqXbAvb4jR48TjxzoVesCxqsHhW_z30HrC5XYN7Q/viewform?usp=header",
    "angry": "https://docs.google.com/forms/d/e/1FAIpQLSfYKzfZ57KYIUVlHy_9zUPZvxByMmJk05jEv98Tu3KP43mZAg/viewform?usp=header",
    "playful": "https://docs.google.com/forms/d/e/1FAIpQLSe6XgIqpRBSKOSSQ5B4-GNSBxhIDC0RrLt3ICUbXpzjPasyKg/viewform?usp=header",
    "sad": "https://docs.google.com/forms/d/e/1FAIpQLSdBGogR1B5F8TWf2lEv2xErK2TwOmYtFZIAzVlbRS21rUb0vQ/viewform?usp=header"
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
is_questionnaire_open = False
questionnaire_launched = False
user_submitted = False 
pause_start_time = 0.0  

pose_ready_time = None
POSE_DISPLAY_DURATION = 10.0


KP = 20.0
KD = 0.0

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def update_headlights(color):
    for geom_id in (headlight_left_id, headlight_right_id):
        mat_id = model.geom_matid[geom_id]
        if mat_id != -1:
            model.mat_rgba[mat_id] = color
        else:
            model.geom_rgba[geom_id] = color

def controller(data, model):
    global current_pose, transition_start_time, pose_ready_time, questionnaire_launched
    
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
                
                # Lancia il questionario SOLO quando la transizione è finita (fase verde)
            if not questionnaire_launched and current_emotion in QUESTIONNAIRE_URLS:
                questionnaire_launched = True
                threading.Thread(target=apri_questionario, args=(current_emotion,), daemon=True).start()
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
        x = (screen_width // 2) + 100  # Corretto per rimanere visibile
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

def change_pose():
    global current_emotion, target_pose, start_pose, pose_count, simulation_finished, transition_start_time, pose_ready_time, questionnaire_launched, user_submitted
    
    # Resetta i flag per la nuova emozione
    emotion_to_question = current_emotion
    questionnaire_launched = False
    user_submitted = False
    pose_ready_time = None
    
    if not pose_queue:
        print("\n[TUTTE LE POSE SONO STATE MOSTRATE]")
        return
    
    start_pose = current_pose.copy()

    # --- LOGICA FARI IN BASE ALL'EMOZIONE ---
    current_emotion = pose_queue.pop()
    target_pose = poses[current_emotion].copy()
    pose_count += 1
    if current_emotion == "sad":
        update_headlights([0.1, 0.0, 0.0, 1.0])
    elif current_emotion == "happy" or current_emotion == "playful":
        update_headlights([0.6, 0.0, 0.8, 1.0])
    elif current_emotion == "angry": 
        update_headlights([1.0, 1.0, 0.0, 1.0])
    
    transition_start_time = time.perf_counter()
    print(f"\r{RED}UN ATTIMO!{RESET}    ", end="", flush=True)
    

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
            
            # Disegna il messaggio sopra il robot (asse Z positivo)
            viewer.user_scn.ngeom = 0
            if transition_start_time is not None:
                disegna_testo(viewer, "Attendi, il robot si sta posizionando!", 0.0, -0.4, 0.2, (1.0, 0.2, 0.2, 1.0))
                
            # --- GESTIONE LUCI SCENA (Spente in transizione) ---
            if transition_start_time is not None:
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

# Dinamico-CL, Luci-on-Random, Closed-World + Questionari + Face Tracking

import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import sys
import termios
import tty
import select
import cv2
import math
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
# LINK QUESTIONARI
# ============================================================
QUESTIONNAIRE_URLS = { 
    "happy": "https://docs.google.com/forms/d/e/1FAIpQLSfwriJ3iiBRoc1hs36jGU75DyYT7zpUTqs58KF7pp38DoDAuA/viewform?usp=header",
    "angry": "https://docs.google.com/forms/d/e/1FAIpQLSdeEgUVh0YpnpmQw6RetrVS9Ln6Vmk7Q3dvV0QqnBtnotJuJA/viewform?usp=header",
    "playful": "https://docs.google.com/forms/d/e/1FAIpQLSck8LZ6QnT_cSfDokIWy_syo9HDnxTYyK2wHhtkd42-8QLl9Q/viewform?usp=header",
    "sad": "https://docs.google.com/forms/d/e/1FAIpQLScN-v70OVrPqL7Qne4uvuJqFyTqxYfvd9VRd2aoWYMtCZSHPA/viewform?usp=header"
}

# ============================================================
# CARICAMENTO MODELLO
# ============================================================
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/Go1_Emotions/scene.xml"
print(f"Caricamento: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.iterations = 200
last_real_time = time.perf_counter()
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
        "duration": 4.0,
        "repeats": 1
    }
}

neutral_stand_pose = np.array([
    0.0,  0.6, -1.5,
    0.0,  0.6, -1.5,
    0.0,  0.9, -2.0,
    0.0,  0.9, -2.0
])

# ============================================================
# STATO E CODA POSE
# ============================================================
phase = "idle"
current_emotion = None
current_animation = None
current_pose = neutral_stand_pose.copy()
start_pose = neutral_stand_pose.copy()
transition_start_time = None
TRANSITION_DURATION = 5.0

anim_elapsed = 0.0
anim_duration = 0.0
anim_keyframes = None
anim_repeats = 1
total_anim_duration = 0.0

last_debug_print = 0.0
last_target_yaw = 0.0
current_hip_rot = 0.0

pose_queue = list(animations.keys())
random.shuffle(pose_queue)
print(f"[INFO] Ordine casuale dei movimenti")

pose_count = 0          
MAX_POSES = 5         
simulation_finished = False  

pose_ready_time = None
POSE_DISPLAY_DURATION = 10.0
is_questionnaire_open = False
pause_start_time = 0.0 
questionnaire_launched = False
user_submitted = False 


KP = 20.0
KD = 0.05

# Variabili Face Tracking
current_yaw = 0.0
target_yaw = 0.0
walk_phase = 0.0
walk_amplitude = 0.0
reference_cx = None

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ============================================================
# CONTROLLER
# ============================================================
def controller(data, model):
    global current_pose, start_pose, transition_start_time, current_animation
    global phase, anim_elapsed, anim_duration, anim_keyframes
    global target_yaw, last_target_yaw, last_real_time, current_hip_rot
    global anim_repeats, total_anim_duration, pose_ready_time, is_questionnaire_open, questionnaire_launched

    if current_animation is None:
        return
     
    current_real_time = time.perf_counter()
    real_dt = current_real_time - last_real_time
    last_real_time = current_real_time

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
        anim_elapsed += real_dt
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

    # --- FACE TRACKING: INCLINAZIONE SPALLE ---
    target_yaw_clamped = max(-0.4, min(0.4, target_yaw))
    hip_error = target_yaw_clamped - current_hip_rot
    
    if abs(target_yaw) < 0.01 and abs(hip_error) < 0.01:
        current_hip_rot = 0.0
    else:
        current_hip_rot += hip_error * 0.02  # Rallentato come richiesto

    dynamic_pose = current_pose.copy()
    dynamic_pose[0] -= current_hip_rot
    dynamic_pose[3] -= current_hip_rot

    # --- CONTROLLO PD ---
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = dynamic_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)

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
# CAMBIO POSA
# ============================================================
def change_pose():
    global phase, current_emotion, current_animation, anim_keyframes, anim_duration
    global start_pose, pose_count, simulation_finished, transition_start_time
    global pose_ready_time, anim_repeats, total_anim_duration, questionnaire_launched, user_submitted
    
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
    
    # Gestione luci in base all'emozione (LUCI-ON-RANDOM)
    if current_emotion == "sad":
        model.geom_rgba[headlight_left_id] = [0.0, 0.0, 1.0, 1.0]      # Blu
        model.geom_rgba[headlight_right_id] = [0.0, 0.0, 1.0, 1.0]
    elif current_emotion in ("happy", "playful"):
        model.geom_rgba[headlight_left_id] = [1.0, 1.0, 1.0, 1.0]      # Bianco
        model.geom_rgba[headlight_right_id] = [1.0, 1.0, 1.0, 1.0]
    elif current_emotion == "angry": 
        model.geom_rgba[headlight_left_id] = [0.0, 1.0, 1.0, 1.0]      # Ciano
        model.geom_rgba[headlight_right_id] = [0.0, 1.0, 1.0, 1.0]
    
    phase = "to_neutral"
    transition_start_time = None

# ============================================================
# CONFIGURAZIONE WEBCAM + YUNET
# ============================================================
yunet_model_path = cv2.samples.findFile('face_detection_yunet_2023mar.onnx')
try:
    face_detector = cv2.FaceDetectorYN.create(yunet_model_path, "", (320, 320), 0.9, 0.3, 5000)
    print("[OK] Modello YuNet caricato correttamente")
except cv2.error:
    print("[ERRORE] Impossibile caricare YuNet.")
    exit(1)
        
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)
print("[OK] Webcam attiva")

cv2.namedWindow('Webcam - Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam - Face Detection', 640, 480)
cv2.moveWindow('Webcam - Face Detection', 50, 50)

VISION_INTERVAL = 0.1
last_vision_time = 0.0

# ============================================================
# INIZIALIZZAZIONE
# ============================================================
old_settings = termios.tcgetattr(sys.stdin)
try:
    tty.setcbreak(sys.stdin.fileno())
    
    data.qpos[7:19] = neutral_stand_pose
    mujoco.mj_forward(model, data)

    change_pose()

    print("\n" + "=" * 50)
    print("  ANIMAZIONI RANDOM + FACE TRACKING + QUESTIONARI")
    print(f"  Mostra {MAX_POSES-1} animazioni casuali")
    print("  Timer automatico: 10 secondi per posa")
    print("=" * 50)

    # ============================================================
    # LOOP PRINCIPALE
    # ============================================================
    glfw.init()
    glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
    
    tracking_activation_time = time.perf_counter() + 3.0
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = 1.4
        viewer.cam.lookat[:] = [0.35, 0.0, 0.35]
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -45.0
        
        while viewer.is_running() and not simulation_finished:
            step_start = time.perf_counter()
            current_time = time.time()
            
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
            
            # --- FACE TRACKING ---
            if current_time - last_vision_time > VISION_INTERVAL:
                last_vision_time = current_time
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, h // 2
                    
                    face_detector.setInputSize((w, h))
                    retval, faces = face_detector.detect(frame)
                    face_detected = isinstance(faces, np.ndarray) and faces.shape[0] > 0
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    
                    if face_detected:
                        x, y, w_face, h_face = faces[0][:4].astype(int)
                        fx = x + w_face // 2
                        
                        if time.perf_counter() < tracking_activation_time:
                            target_yaw = 0.0
                            if reference_cx is None:
                                reference_cx = fx
                        else:
                            if reference_cx is None:
                                reference_cx = fx
                                
                            offset = fx - reference_cx
                            
                            if abs(offset) <= 50:
                                target_yaw = 0.0
                            else:
                                target_yaw = -(offset / 50.0) * 0.12
                            
                            cv2.line(frame, (cx, cy), (fx, cy), (0, 255, 0), 2)
                            cv2.circle(frame, (fx, cy), 4, (0, 255, 0), -1)
                            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
                    else:
                        target_yaw = 0.0
                    
                    cv2.imshow('Webcam - Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # --- MESSAGGIO NELLA SCENA 3D ---
            viewer.user_scn.ngeom = 0
            disegna_testo(viewer, "Lui ti sta guardando...", -0.3, -1.0, 0.7)
            
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
            
            # --- FISICA E RENDERING (SOLO UNA VOLTA) ---
            controller(data, model)
            
            # --- TIMING LOOP ---
            time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            mujoco.mj_step(model, data)
            viewer.sync()

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    cap.release()
    cv2.destroyAllWindows()

print("\nSimulazione terminata.")

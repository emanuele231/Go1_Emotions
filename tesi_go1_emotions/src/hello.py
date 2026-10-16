import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
import time

# ============================================================
# 1. CONFIGURAZIONE MUJOCO
# ============================================================
model_path = "/home/emanuele/mujoco_menagerie/unitree_go1/Go1_Emotions/scene.xml"
print(f"Caricamento modello MuJoCo: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.iterations = 200

# === POSE ===
base_pose = np.array([
    0.0,  1.3, -2.3,   # Front Right (Seduto)
    0.0,  1.3, -2.3,   # Front Left
    0.0,  1.3, -2.3,   # Rear Right
    0.0,  1.3, -2.3    # Rear Left
])

ready_pose = np.array([
    0.0, -0.5, -1.5,   # Front Right: SOLLEVATA (piegata in alto)
    0.0,  0.9, -1.5,   # Front Left: a terra, posizione neutra
    0.0,  0.9, -1.5,   # Rear Right: a terra, posizione neutra
    0.0,  0.9, -1.5    # Rear Left: a terra, posizione neutra
])

# Variabili di stato
current_state = "base"
transition_duration = 1.0
paw_transition_duration = 0.1
transition_start_time = None
start_pose = base_pose.copy()
target_pose = base_pose.copy()
interpolated_pose = base_pose.copy()

READY_DURATION = 15.0       # Durata del saluto in secondi
ready_enter_time = None 
waiting_for_hand_down = False

# DEBOUNCE
GESTURE_CONFIRMATION_TIME = 1.5  
hand_up_start_time = None 

start_height = 0.0
target_height = 0.0
current_height = 0.0

# Parametri di controllo
KP = 20.0  
KD = 0.0   

# ============================================================
# 2. CONFIGURAZIONE MEDIAPIPE
# ============================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("Inizializzazione webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)

print("=" * 50)
print("GESTURE STAND UP - Cinematica Inversa")
print("I piedi resteranno INCOLLATI a terra durante il movimento.")
print("Alza UNA SOLA mano per far alzare il robot!")
print("=" * 50)

# ============================================================
# 3. FUNZIONI CINEMATICHE E DI CONTROLLO
# ============================================================
def smoothstep(t):
    return t * t * (3 - 2 * t)

def calculate_kinematic_height(pose):
    """
    Calcola l'altezza cinematica del torso per mantenere i piedi a terra.
    """
    L_thigh = 0.215
    L_calf = 0.215
    
    # Usa la zampa anteriore sinistra (indice 4, 5) che resta sempre a terra
    thigh_angle = pose[4]
    calf_angle = pose[5]
    
    h = L_thigh * np.cos(thigh_angle) + L_calf * np.cos(thigh_angle + calf_angle)
    
    FOOT_OFFSET = 0.03
    
    return h + FOOT_OFFSET

def apply_target_state(new_state, is_timeout=False):
    global current_state, start_pose, target_pose, transition_start_time
    global start_height, target_height, current_height
    global ready_enter_time, waiting_for_hand_down
    
    if current_state == new_state:
        return
    
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> Transizione: {current_state.upper()} -> {new_state.upper()}")
    
    start_pose = interpolated_pose.copy()
    start_height = current_height
    
    if new_state == "ready":
        target_pose = ready_pose.copy()
        ready_enter_time = time.time()
        waiting_for_hand_down = False  # Reset del lockout
        
    else:
        target_pose = base_pose.copy()
        ready_enter_time = None
        if is_timeout:
            waiting_for_hand_down = True  # Attiva il lockout 
    
    target_height = calculate_kinematic_height(target_pose)
    
    current_state = new_state
    transition_start_time = time.time()

def pd_controller(data, model, target_pose):
    """Controllo PD con damping alto per smorzare oscillazioni"""
    for i in range(model.nu):
        current_q = data.qpos[7 + i]
        current_dq = data.qvel[6 + i]
        target_q = target_pose[i]
        data.ctrl[i] = KP * (target_q - current_q) + KD * (0.0 - current_dq)
# ============================================================
# 4. LOOP PRINCIPALE
# ============================================================
# Setup iniziale: calcoliamo l'altezza corretta per la base_pose
initial_height = calculate_kinematic_height(base_pose)
print(f"[INFO] Altezza cinematica iniziale: {initial_height:.3f}m")

data.qpos[2] = initial_height
data.qpos[7:19] = base_pose
current_height = initial_height
start_height = initial_height
target_height = initial_height

mujoco.mj_forward(model, data)

HAND_UP_THRESHOLD = 0.4

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.2]
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while viewer.is_running() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- MEDIAPIPE ---
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            gesture_detected = False
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                landmarks = results.pose_landmarks.landmark
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                if left_wrist.y < HAND_UP_THRESHOLD or right_wrist.y < HAND_UP_THRESHOLD:
                    gesture_detected = True
                
                # --- DEBUG VISIVO ---
                h, w, _ = image.shape
                threshold_y_pixel = int(HAND_UP_THRESHOLD * h)
                cv2.line(image, (0, threshold_y_pixel), (w, threshold_y_pixel), (0, 255, 255), 2)
                
                status_color = (0, 255, 0) if current_state == "ready" else (0, 0, 255)
                
                # Mostra stato a video con countdown di conferma
                if current_state == "ready" and ready_enter_time is not None:
                    elapsed = time.time() - ready_enter_time
                    remaining = max(0, READY_DURATION - elapsed)
                    status_text = f"SALUTO: {remaining:.1f}s rimanenti"
                    status_color = (0, 255, 0)
                    
                elif hand_up_start_time is not None and not waiting_for_hand_down:
                    # Mano alzata ma in attesa di conferma
                    elapsed_confirm = time.time() - hand_up_start_time
                    remaining_confirm = max(0, GESTURE_CONFIRMATION_TIME - elapsed_confirm)
                    status_text = f"Conferma saluto: {remaining_confirm:.1f}s / {GESTURE_CONFIRMATION_TIME}s"
                    status_color = (0, 255, 255)  # Giallo
                    
                elif waiting_for_hand_down:
                    status_text = "LOCKOUT: Abbassa la mano"
                    status_color = (0, 0, 255)  # Rosso
                    
                else:
                    status_text = "Pronto (alza la mano per 2s)"
                    status_color = (255, 255, 255)  # Bianco
                
                cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow('Gesture Stand Up Control', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            # Gestione del timer della mano alzata
            if gesture_detected:
                if hand_up_start_time is None:
                    hand_up_start_time = time.time()
            else:
                # Se la mano scende prima dei 2 secondi, resetta il timer
                hand_up_start_time = None

            #  Macchina a stati
            if current_state == "ready":
                # Se siamo in saluto, controlliamo solo il timer dei 3 secondi
                if ready_enter_time is not None:
                    elapsed_ready = time.time() - ready_enter_time
                    if elapsed_ready >= READY_DURATION:
                        apply_target_state("base", is_timeout=True)
                        
            else:
                # Se siamo in base, controlliamo il lockout e il debounce
                if waiting_for_hand_down:
                    # Lockout attivo: aspettiamo che la mano si abbassi
                    if not gesture_detected:
                        waiting_for_hand_down = False
                        print("[INFO] Mano abbassata. Lockout disattivato.")
                else:
                    # Nessun lockout: controlliamo se la mano è alzata da 2 secondi
                    if hand_up_start_time is not None:
                        elapsed_hand_up = time.time() - hand_up_start_time
                        
                        # Se sono passati 2 secondi e non siamo già in transizione
                        if elapsed_hand_up >= GESTURE_CONFIRMATION_TIME and transition_start_time is None:
                            print(f"\n[{time.strftime('%H:%M:%S')}] >>> Gesto confermato! Inizio saluto.")
                            apply_target_state("ready")
                            hand_up_start_time = None  # Reset per evitare ri-trigger
            
            # --- MUJOCO ---
            if transition_start_time is not None:
                elapsed = time.time() - transition_start_time

                t_body = min(elapsed / transition_duration, 1.0)
                t_paw = min(elapsed / paw_transition_duration, 1.0)
                
                t_smooth_body = smoothstep(t_body)
                t_smooth_paw = smoothstep(t_paw)

                interpolated_pose = start_pose + t_smooth_body * (target_pose - start_pose)

                interpolated_pose[0:3] = start_pose[0:3] + t_smooth_paw * (target_pose[0:3] - start_pose[0:3])
                if t_body >= 1.0:
                    transition_start_time = None
            current_height = calculate_kinematic_height(interpolated_pose)
            
            pd_controller(data, model, interpolated_pose)
            
            data.qpos[2] = current_height
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time.sleep(model.opt.timestep)

cap.release()
cv2.destroyAllWindows()
print("\nSimulazione terminata.")

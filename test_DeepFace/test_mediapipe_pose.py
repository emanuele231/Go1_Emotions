import cv2
import mediapipe as mp
import time

# Inizializza MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("Avvio MediaPipe Pose (Rilevamento Corpo)...")
print("Premi 'q' per uscire.")
print("=" * 40)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)

current_state = None

# Configura il rilevatore (soglia di confidenza)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe richiede immagini in RGB, OpenCV usa BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Torna a BGR per mostrare il video a schermo
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Controlla se è stato rilevato un corpo
        if results.pose_landmarks:
            new_state = "ready"
            
            # Disegna lo scheletro a video
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            # ESEMPIO: Leggiamo le coordinate del Polso Destro (indice 16)
            # Le coordinate sono normalizzate da 0.0 a 1.0 (0.0 è in alto/sinistra, 1.0 in basso/destra)
            right_wrist = results.pose_landmarks.landmark[16]
            # right_shoulder = results.pose_landmarks.landmark[12]
            
            # Stampa le coordinate nel terminale (solo per debug)
            # print(f"Polso Destro -> X: {right_wrist.x:.2f} | Y: {right_wrist.y:.2f} | Z: {right_wrist.z:.2f}")
            
        else:
            new_state = "sad"
            
        # Logica anti-spam per il terminale
        if new_state != current_state:
            timestamp = time.strftime('%H:%M:%S')
            if new_state == "ready":
                print(f"[{timestamp}] im ready! (Corpo rilevato)")
            else:
                print(f"[{timestamp}] sad (Nessun corpo)")
            current_state = new_state
            
        cv2.imshow('MediaPipe Pose - Body Tracking', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Script terminato.")

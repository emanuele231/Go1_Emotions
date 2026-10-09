import cv2
import time

# Usiamo il file XML che abbiamo scaricato manualmente
cascade_path = "/home/emanuele/unitree_sdk2_python/venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("[ERRORE] Impossibile caricare il classificatore Haar. Controlla il percorso.")
    exit(1)

print("Rilevamento presenza volto avviato.")
print("Premi 'q' per uscire.")
print("=" * 40)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)
    
ABSENCE_THRESHOLD = 5.0
PRESENCE_THRESHOLD = 2.5
last_face_seen_time = time.time()

current_state = "sad"
last_face_seen_time = time.time()
face_start_time = None



while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    current_time = time.time()
    
    # OpenCV rileva i volti meglio in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Rileva i volti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_detected = len(faces) > 0
    # Determina il nuovo stato
    if face_detected:
        last_face_seen_time = current_time
        if face_start_time is None:
            face_start_time = current_time
    else:
        face_start_time = None
        
    # --- MACCHINA A STATI ---
    new_state = current_state
    
    # Logica per passare a "ready"
    if current_state == "sad" and face_start_time is not None:
        time_being_seen = current_time - face_start_time
        if time_being_seen >= PRESENCE_THRESHOLD:
            new_state = "ready"
            
    # Logica per tornare a "sad"
    elif current_state == "ready":
        time_since_last_face = current_time - last_face_seen_time
        if time_since_last_face >= ABSENCE_THRESHOLD:
            new_state = "sad"
            face_start_time = None 
            
    # Stampa il messaggio SOLO se lo stato è cambiato (anti-spam)
    if new_state != current_state:
        timestamp = time.strftime('%H:%M:%S')
        if new_state == "ready":
            print(f"[{timestamp}] im ready! (Volto confermato per {PRESENCE_THRESHOLD}s)")
        else:
            print(f"[{timestamp}] sad (Nessun volto da {ABSENCE_THRESHOLD}s)")
        current_state = new_state
        
    # Disegna un rettangolo blu intorno alla faccia (utile per debug)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow('Face Presence Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Script terminato.")

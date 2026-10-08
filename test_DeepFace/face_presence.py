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

current_state = None  # Tiene traccia dello stato attuale per non spammare il terminale

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # OpenCV rileva i volti meglio in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Rileva i volti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Determina il nuovo stato
    if len(faces) > 0:
        new_state = "ready"
    else:
        new_state = "sad"
        
    # Stampa il messaggio SOLO se lo stato è cambiato
    if new_state != current_state:
        timestamp = time.strftime('%H:%M:%S')
        if new_state == "ready":
            print(f"[{timestamp}] im ready!")
        else:
            print(f"[{timestamp}] sad")
        current_state = new_state
        
    # Disegna un rettangolo blu intorno alla faccia (utile per debug)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    cv2.imshow('Face Presence Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Script terminato.")

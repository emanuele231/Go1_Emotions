import cv2   #Serve solo ad accendere la webcam e mostrare il video a schermo.
from deepface import DeepFace
import time

print("=" * 50)
print("RICONOSCIMENTO EMOZIONI CON DEEPFACE")
print("=" * 50)

# Apri la webcam (0 = webcam integrata)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam")
    exit(1)

print("[OK] Webcam attiva")
print("Premi 'q' per uscire")
print("=" * 50)

last_analysis_time = 0
analysis_interval = 2  # Analisi ogni 2 secondi

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mostra il video della webcam
    cv2.imshow('DeepFace - Riconoscimento Emozioni', frame)
    
    # Analisi periodica (non ogni frame, DeepFace è lento)
    current_time = time.time()
    if current_time - last_analysis_time > analysis_interval:
        try:
            # Analizza il frame, Diciamo a DeepFace di cercare solo le emozioni (potrebbe anche cercare età, genere, razza, ma a noi non serve).
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False, #Se per un attimo esci dall'inquadratura o ti giri, di solito DeepFace manderebbe un errore e chiuderebbe il programma. Con questo flag a False, se non vede una faccia, restituisce semplicemente un risultato nullo senza crashare.
                silent=True  # Riduce l'output verboso
            ) #a variabile result è un dizionario Python che contiene il "verdetto" della rete neurale. (in questa riga viene fatta la classificazione con migliaia di immagini di volti
            
            # DeepFace può restituire una lista o un dizionario
            if isinstance(result, list):
                result = result[0]
            
            # Estrai informazioni (rileviamo solo il sorriso, tutto il resto e neutro)
            dominant = result['dominant_emotion']
            if dominant == 'happy':
                print(f"\n[{time.strftime('%H:%M:%S')}] >>> STATO: FELICE (Happy) <<<")
            else:
                print(f"\n[{time.strftime('%H:%M:%S')}] STATO: Neutro/Altro ({dominant})")
            
            last_analysis_time = current_time
            
        except Exception as e:
            print(f"[ERRORE] {e}")
    
    # Esci con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pulizia
cap.release()
cv2.destroyAllWindows()
print("\nScript terminato.")

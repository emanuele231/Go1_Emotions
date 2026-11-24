import cv2

# Prova gli indici da 0 a 3
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Telecamera trovata all'indice: {i}")
        ret, frame = cap.read()
        if ret:
            print(f"  -> Risoluzione: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    else:
        print(f"Indice {i}: Non disponibile")

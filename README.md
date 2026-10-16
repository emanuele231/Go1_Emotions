# Unitree Go1 

<p float="left">
  <img src="go1.png" width="400">
</p>

## 1- UniTree Go1 panoramica

### L'App Ufficiale Unitree (iOS / Android)
È l'interfaccia primaria e più user-friendly. Si connette al robot tramite Wi-Fi diretto o hotspot.

**Modalità Sport (Sport Mode)**: È il "cuore" del Go1. Con un tocco, il robot si alza da terra e si bilancia automaticamente.
Da qui in poi, l'utente non controlla i singoli motori, ma dà comandi di direzione (avanti, indietro, ruota) tramite un joystick virtuale.
**Comandi "Trick" preimpostati**: L'app invia comandi ad alto livello come Sit Down (Seduto), Lie Down (Sdraiato), Recovery Stand (Rialzati se caduto) o Dance. 
Questi comandi sono gestiti dal Sport Client interno al robot, che utilizza algoritmi complessi (MPC - Model Predictive Control) 
per calcolare in millisecondi come muovere i 12 motori senza farlo cadere. In simulazione, noi dobbiamo "fingere" di essere questo Sport Client.

### Il Telecomando Fisico (Remote Controller)
È il gamepad (simile a quello di una Xbox) che si accoppia via 2.4GHz. È lo strumento che hai usato per i primissimi test.
**Controllo Tattile**: Offre una precisione che il touchscreen non ha, fondamentale per movimenti fini o per testare la reattività del robot.
**Pulsanti Dedicati**: Ha tasti fisici per le funzioni critiche: alzarsi, sdraiarsi e, soprattutto, il blocco di emergenza 
o il ripristino immediato della postura se il robot perde l'equilibrio.
**Sicurezza**: È il metodo più affidabile per riprendere il controllo manuale se il software personalizzato (SDK) 
dovesse inviare comandi errati e far cadere il robot.

### I Sensori e il "Tracker" Interno (Propriocezione)
Il Go1 non è un'auto radiocomandata; sa sempre dove si trova nello spazio grazie ai suoi sensori interni:
**IMU (Inertial Measurement Unit)**: Un giroscopio e accelerometro ad alta frequenza che dice al robot il suo beccheggio (pitch), 
rollio (roll) e imbardata (yaw). È il senso dell'equilibrio del robot.
**Encoder dei Giunti**: Ogni uno dei 12 motori sa esattamente a quale angolo si trova e con quale velocità sta ruotando.
**Telecamera di Profondità** (opzionale ma comune): Sul muso, usata dall'app per l'evitamento ostacoli e per la mappatura dell'ambiente.
Il nostro script MuJoCo legge data.qpos (posizione) e data.qvel (velocità) per emulare esattamente i dati che questi sensori fisici inviano al cervello del robot.


## 2 - Il Ponte verso il Codice: Unitree SDK (C++ e Python)
Qui è dove l'utente diventa sviluppatore. L'SDK (Software Development Kit) di Unitree si divide in due livelli di controllo fondamentali:
**High-Level Control (Sport Mode API)**:
Tu invii comandi astratti come client.StandUp(), client.SitDown(), o client.SetWalkSpeed(0.5, 0, 0).
Il computer di bordo del Go1 fa tutto il lavoro pesante di bilanciamento. Questo è il metodo più sicuro e consigliato per il tuo progetto.
**Low-Level Control (Motor API)**:
  Tu invii comandi diretti a ogni singolo motore: Posizione (P), Velocità (V) e Coppia/Torque (T). 
  È esattamente quello che abbiamo simulato con il nostro controllore PD (KP * errore + KD * velocità). È potentissimo, 
  ma se sbagli i valori, il robot si ribalta o danneggia i motori.
  
## 3 - Il Motore Fisico e l'Ambiente di Simulazione

### MuJoCo (Multi-Joint dynamics with Contact)
È il cuore della simulazione. Sviluppato originariamente per la ricerca in robotica e ora mantenuto da Google 
DeepMind, è un motore fisico estremamente veloce e preciso. 
Cosa fa: Calcola in tempo reale la gravità, l'attrito, le collisioni e le forze che agiscono sui 12 motori (giunti) del Go1.
Perché lo usiamo: Ci permette di testare pose, transizioni e il controllo PD (Proporzionale-Derivativo) in totale sicurezza, senza r
ischiare di rompere il robot fisico.

**MuJoCo Menagerie**
È la libreria ufficiale dei modelli robotici curata da DeepMind.
Fornisce i file .xml (come il nostro scene.xml) che descrivono la geometria, le masse, i limiti dei giunti e i colori del robot.
Abbiamo usato il modello Unitree Go1, già calibrato con i pesi e le dimensioni reali del cane robot.

**mujoco.viewer**
È l'interfaccia grafica (basata su GLFW) che ci permette di vedere la simulazione in 3D, muovere la telecamera
e, soprattutto, intercettare gli input da tastiera per i test manuali.

## 4 - Lo Stack di Visione e Intelligenza Artificiale

### OpenCV (opencv-python e opencv-contrib-python)
È la libreria standard per la computer vision.
 Gestisce l'apertura della webcam, cattura i frame video e li converte in scala di grigi.
**Haar Cascades**: Abbiamo usato i classificatori Haar (file .xml di OpenCV) per il rilevamento veloce e leggero dei volti. 
È il primo filtro che decide se c'è qualcuno davanti al robot.

**DeepFace**
È un framework Python che semplifica l'uso di reti neurali complesse per l'analisi del volto.
Analizza il volto rilevato da OpenCV e restituisce un dizionario con le percentuali delle emozioni (es. happy: 85.4%, sad: 2.1%).
Sotto il cofano, DeepFace usa TensorFlow e tf-keras per eseguire le reti neurali (come VGG-Face o Facenet) che estraggono i tratti somatici.

**MediaPipe**
Sviluppata da Google, è una libreria per il tracciamento del corpo e delle mani in tempo reale.
Ruolo nel progetto: Anche se nello script finale ci siamo concentrati sul volto, MediaPipe è stata installata e testata per rilevare i
33 punti chiave del corpo umano. Sarà utilissima in futuro se vorremo far imitare al robot i movimenti delle braccia o della postura dell'utente.

### Lo Stack Matematico e le Dipendenze "Nascoste"
Queste librerie sono il collante che tiene insieme tutto il sistema.
**NumPy (numpy)**
È la base del calcolo scientifico in Python.
Gestisce gli array multidimensionali. Le pose del robot (es. sad_pose) sono array NumPy a 12 dimensioni.
L'interpolazione smoothstep e le operazioni matriciali del controllo PD si basano interamente su di esso.

**TensorFlow (tensorflow)** e **tf-keras**
Sono i motori di calcolo per il machine learning.
Sono richiesti da DeepFace per caricare ed eseguire i modelli di riconoscimento facciale. 
(Nota: li usiamo in modalità CPU, dato che non abbiamo GPU dedicate in questo ambiente).

**Protobuf (protobuf)**
È una libreria di Google per la serializzazione dei dati strutturati.
Il problema risolto: È stata la causa principale del nostro "inferno delle dipendenze". TensorFlow, MediaPipe e OpenCV 
richiedevano versioni di Protobuf incompatibili tra loro. Abbiamo dovuto trovare la versione ponte (4.25.3) per farle convivere.

## 5 - Sistema Emozioni (Emotion Vision System)

Il robot Go1 implementa un sistema di emozioni reattivo basato su visione artificiale, 
che combina rilevamento della presenza (Haar Cascade) e riconoscimento delle emozioni facciali (DeepFace). 
Il sistema gestisce tre stati emotivi con transizioni fluide e comportamenti distintivi.

### Architettura del Sistema
Il sistema opera su due livelli di rilevamento:

    Haar Cascade (10Hz): Rilevamento rapido della presenza di un volto
    DeepFace (0.5Hz): Analisi dell'emozione facciale ogni 1.5 secondi

### Struttura attuale del sistema emotivo
Il sistema trasforma il robot da una semplice macchina a un compagno reattivo e "vivente", capace di
osservare e rispondere all'umore umano. 
Il robot inizia il suo ciclo in stato di riposo **(SAD)**: è sdraiato a terra, con il corpo abbassato e i fari spenti, 
simulando un cane rilassato o malinconico. 

Non appena la telecamera percepisce la presenza costante di una persona, il robot si alza con 
un movimento fluido e graduale, passando allo stato attento **(READY)**. In questa fase, assume una postura eretta e 
accende i fari frontali di luce bianca, segnalando che ha notato l'utente.

Se l'interazione si fa positiva e la persona sorride, il robot esplode di gioia **(HAPPY)**: con uno scatto rapido ed energico, 
esegue un saltino allegro e tinge i suoi fari di verde brillante. 
Quando il sorriso svanisce, il robot si calma dolcemente, tornando allo stato attento. Infine, se la persona si allontana,
il sistema non reagisce in modo nervoso: attende qualche secondo per poi "accasciarsi" lentamente a terra, tornando allo stato di riposo. 
Grazie a un'interpolazione matematica che sincronizza perfettamente l'estensione delle zampe con l'altezza del torso, 
ogni transizione è priva di scatti meccanici. Il risultato è un robot con una personalità canina, empatica e incredibilmente naturale.

L'intenzione del **saluto** è creare un'interazione naturale ed espressiva, simile a un cane che dà la zampa, 
mantenendo però un controllo rigoroso per evitare comportamenti robotici nervosi o loop infiniti. 
Il sistema è progettato per richiedere che l'utente tenga la mano alzata per due secondi consecutivi prima di attivare il gesto,
filtrando così i movimenti accidentali o i passaggi veloci. Una volta confermato, il robot solleva una zampa anteriore in modo scattante mentre il corpo si assesta dolcemente, garantendo attraverso il calcolo cinematico che le altre tre zampe rimangano perfettamente incollate a terra senza scivolare. 
Il saluto ha una durata fissa e limitata di tre secondi, dopodiché il robot torna autonomamente alla posa di riposo. 
Per evitare ripetizioni incontrollate, si attiva immediatamente un blocco di sicurezza che ignora la mano se questa rimane alzata,
richiedendo all'utente di abbassarla e rialzarla consapevolmente per initiare un nuovo saluto. In questo modo il robot appare educato, reattivo e fisicamente stabile.


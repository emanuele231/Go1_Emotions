# Unitree Go1 

## Sistema Emozioni (Emotion Vision System)
Il robot Go1 implementa un sistema di emozioni reattivo basato su visione artificiale, 
che combina rilevamento della presenza (Haar Cascade) e riconoscimento delle emozioni facciali (DeepFace). 
Il sistema gestisce tre stati emotivi con transizioni fluide e comportamenti distintivi.

### Architettura del Sistema
Il sistema opera su due livelli di rilevamento:

    Haar Cascade (10Hz): Rilevamento rapido della presenza di un volto
    DeepFace (0.5Hz): Analisi dell'emozione facciale ogni 1.5 secondi

<p float="left">
  <img src="go1.png" width="400">
</p>



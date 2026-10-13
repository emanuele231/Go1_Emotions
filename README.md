# Unitree Go1 

<p float="left">
  <img src="go1.png" width="400">
</p>

## Sistema Emozioni (Emotion Vision System)
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


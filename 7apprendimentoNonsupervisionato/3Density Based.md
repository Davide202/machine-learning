Vediamo un ulteriore metodo per il clustering conosciuto come
Density Based Special Clustering ... (DBSCAN).

Modelli visti fin ora:

K-Means: il numero di cluster va definito a priori

Clustering Gerarchico: il numero di cluster va definito a posteriori

DBSCAN: non serve definire il numero di cluster.

Parametri del DBSCAN:

- eps che definisce la distanza massima tra due osservazioni nello stesso vicinato

- minPts che definisce il numero minimo di osservazioni richieste per formare un cluster (>= numDims + 1, min=3)

Vediamo passo passo come funziona il DBSCAN:

- Scelgo i valori di eps e minPts

- Per ogni osservazione:
    
    Ci sono più minPts osservazioni in un raggio di distanza eps dall'osservazione?
    
    SI -> l'osservazione è un 'core point' e forma un cluster 
    
    NO ->  c'è un core point nel raggio di distanza eps dall'osservazione?
            
            SI -> l'osservazione è un 'border point' e viene assegnata al cluster rappresentato dal core point
            
            NO > l'osservazione è un 'noise point' e non viene assegnata ad alcun cluster.

Vantaggi del DBSCAN:
non serve definire il numero di cluster
è resistente agli outlier
non limita i cluster ad una forma sferica questo accade quando si trovano due core point entro la distanza eps,
infatti i cluster di questi core points vengono fusi tra di loro.

Anomaly Detection
vista la sua capacità di individuare il rumore nei dati
il DBSCAN è particolarmente adatto ad attività di Anomaly Detection,
è una tecnica utilizzata per riconoscere pattern inusuali che non sono conformi al comportamento atteso.
Nel machine learning questi pattern si presentano sottoforma di outlier,
infatti l'anomaly detection è conosciuto anche come outlier detection.


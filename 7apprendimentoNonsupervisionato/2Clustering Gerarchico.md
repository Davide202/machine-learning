Clustering Gerarchico

Esistono due tipologie:
1. Agglomerato (bottom up)
    Partiamo con un cluster per ogni singola osservazione e li raggruppiamo fino ad
    avere un unico grande cluster.
2. Divisivo (top down)
    Partiamo con tutti gli elementi in un singolo cluster e lo suddividiamo
    fino ad avere un cluster per ogni osservazione.

L'agglomerativo è molto più utilizzato ed è quello sul quale ci focalizziamo.
Durante il processo di accorpamento dei cluster dobbiamo creare una memoria
in grado di salvare ogni step che abbiamo compiuto per raggruppare i cluster,
possiamo creare questa memoria creando un Dendogramma.
Fissando un diverso valore di soglia abbiamo diversi numeri di cluster.

Per determinare il valore di soglia ideale esistono diverse tecniche,
una ad esempio è il Coefficiente di inconsistenza che ci dice 
di posizionare la soglia sul cluster la cui altezza sul Dendogramma è notevolmente
maggiore rispetto all'altezza media dei cluster sottostanti.

Vantaggi e Svantaggi Clustering Gerarchico

Il vantaggio più grande è che permette di eseguire il clustering
senza definire a priori il numero di cluster.

Lo svantaggio è che è dispendioso in termini di risorse di calcolo. 


Che metriche utilizzare per scegliere quali cluster unire

Single Linkage 
    unire i cluster con la minima distanza tra le due osservazioni più vicine.
Complete Linkage
    unire i cluster con la minima distanza tra le due osservazioni più lontane.
Average Linkage
    unire i due cluster in cui la media della distanza tra tutte le osservazioni di due cluster è minore.
Ward's Linkage
    unire i clusters che uniti hanno la minore somma dei quadrati delle distanze.

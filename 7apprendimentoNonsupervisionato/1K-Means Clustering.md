
---

# Il Clustering e l'Algoritmo K-Means

## Introduzione al Clustering

Il Clustering è un problema tipico dell'**apprendimento non supervisionato** (unsupervised learning). L'obiettivo è individuare strutture interne o pattern nei dati senza avere a disposizione le **label** (etichette), ovvero senza conoscere a priori l'appartenenza di una informazione a una specifica classe.

> **OSS:** Sostanzialmente, stiamo chiedendo all'algoritmo di cercare e creare in autonomia delle "classi" (cluster) raggruppando i nostri dati in base alle loro somiglianze.

## Come funziona il K-Means Clustering

Il K-Means è in assoluto l'algoritmo di clustering più diffuso, performante e semplice da implementare. Il suo funzionamento segue questi step:

1. **Scelta di K:** Definisco a priori il numero $K$ di cluster che voglio creare.
2. **Inizializzazione:** Seleziono casualmente $K$ punti nel dataset che fungeranno da centroidi iniziali.
3. **Assegnazione:** Calcolo la distanza tra ogni singolo centroide e tutte le osservazioni nel dataset. Assegno quindi ogni osservazione al cluster rappresentato dal centroide più vicino.
4. **Aggiornamento:** Ricalcolo la posizione dei centroidi prendendo la media (il centro di massa) di tutti gli esempi che sono finiti in quel cluster.
5. **Iterazione:** Ripeto i punti 3 e 4. L'algoritmo si ferma (converge) quando, ricalcolando i centroidi, nessun esempio cambia più il cluster di appartenenza.

## Come scegliere il numero di Cluster (Valore di K)

Poiché non abbiamo etichette, non possiamo calcolare un'accuratezza standard. Costruiamo invece diversi modelli con diversi valori di $K$ e ne confrontiamo le performance utilizzando la metrica **SSD** (Sum of Squared Distances), nota in Scikit-Learn anche come **Inertia** o **WCSS** (Within-Cluster Sum of Squares).

L'SSD calcola la somma delle distanze al quadrato tra i centroidi e ogni osservazione che appartiene al loro cluster, sommando poi i valori di tutti i cluster:

$$SSD = \sum_{j=1}^{K} \sum_{i=1}^{n_j} (x_i - c_j)^2$$

Ovviamente, il valore dell'SSD andrà fisiologicamente a scendere man mano che aumentiamo il numero di cluster $K$ (se $K$ fosse uguale al numero totale di esempi nel dataset, l'SSD sarebbe 0, ma il modello sarebbe inutile).

### Il Metodo del Gomito (Elbow Method)

Per determinare il valore corretto di $K$ senza cadere nell'overfitting, utilizziamo questo metodo grafico:

* Si traccia un grafico inserendo il numero di cluster $K$ sull'asse X e il valore della metrica SSD sull'asse Y.
* Osservando la curva decrescente, si immagina che il grafico rappresenti un braccio umano.
* Si va alla ricerca del **gomito**, ovvero il punto di flesso in cui la curva smette di scendere in modo drastico (forte pendenza) e inizia ad appiattirsi, decrescendo di poco ad ogni nuovo cluster aggiunto.
* Quel valore di $K$ rappresenta il miglior compromesso tra un basso errore di assegnazione e una giusta complessità del modello.

---

## Integrazioni Fondamentali (Best Practices)

Per usare il K-Means con successo su dati reali, tieni a mente questi tre concetti che completano la teoria:

* **Standardizzazione dei dati (Scaling):** Poiché il K-Means si basa sul calcolo geometrico delle distanze (solitamente la distanza Euclidea), è **obbligatorio** scalare i dati prima di addestrare il modello. Se non lo fai, le feature con valori numerici naturalmente più grandi domineranno matematicamente la creazione dei cluster.
* **Il problema dell'inizializzazione casuale:** Il K-Means classico sceglie i centroidi in modo totalmente casuale al punto 2, rischiando di farli partire vicinissimi e creare cluster sub-ottimali. Oggi si usa la variante **K-Means++** (che è quella predefinita in Scikit-Learn), la quale sceglie i centroidi iniziali il più distanti possibile tra loro, garantendo risultati migliori e più veloci.
* **La metrica Silhouette (Oltre l'Elbow):** A volte il grafico del gomito è "dolce" e non mostra uno spigolo netto, rendendo difficile scegliere $K$. In questi casi si affianca il **Silhouette Score**, una metrica tra -1 e 1 che valuta non solo quanto i dati siano vicini al proprio centroide (coesione), ma anche quanto siano lontani dai cluster vicini (separazione). Valori vicini a 1 indicano cluster perfetti.


---
APPUNTI ORIGINALI
Il Clustering è un problema tipico dell'apprendimento non supervisionato in cui 
dobbiamo individuare strutture interne ai dati senza avere a disposizione degli esempi,
cioè delle label che rappresentano l'appartenenza di una informazione ad una classe.

Il K-Means Clustering è in assoluto l'algoritmo di clustering più diffuso, performante e semplice da implementare, funziona così:
1. Scelgo il numero di cluster K da creare
2. Seleziono casualmente K centroidi
3. Calcolo la distanza tra ogni centroide e tutte le osservazioni
4. Assegno le osservazioni al cluster rappresentato dal centroide più vicino
5. Ricalcolo i centroidi come la media degli esempi per ogni cluster
6. Ripeto dal punto 2 fino a quando nessun esempio cambia più cluster.

OSS: Sostanzialmente stiamo cercando delle "classi" per i nostri dati.

Come scegliere il numero di Clusters (valore di K)?

Costruiamo diversi modelli con diversi valori di K e li confrontiamo (SSD).

SSD = la somma delle distanze al quadrato tra i centroidi ed ogni osservazione che appartiene al cluster e sommiamo i valori ottenuti per ogni centroide.

Ovviamente il valore andrà a scendere man mano che aumentiamo il numero di cluster.

Per determinare il valore corretto di K, possiamo utilizzare un metodo grafico: ELBOW METHOD,
(TODO scrivere come funziona questo metodo)
che consiste nell'identificare il valore del "gomito" supponento che il grafico rappresenta un braccio.
Sostanzialmente è il minimo valore di K per cui la curva rappresentata da (K,SSD) decresce "poco".






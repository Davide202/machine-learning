# %%
import pandas as pd
import os

iris_path = "../../data/iris.csv"

# Stampa a schermo il percorso della cartella di lavoro corrente da cui stai eseguendo lo script
print(os.getcwd())


# Carica il file CSV in un DataFrame Pandas. 
# header=None indica che il file non ha intestazioni originali valide.
# names=[...] sovrascrive/imposta manualmente i nomi delle colonne.
iris = pd.read_csv(
    iris_path, 
    header=None, 
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)

# %%
# Mostra le PRIME 10 righe (Senza print!)
iris.head(10)

# %%
# Mostra la struttura del DataFrame
iris.info()

# %%
# Estrae una singola colonna dal DataFrame. Il risultato è una "Series" (una struttura dati monodimensionale).
Y = iris['species']
# Mostra la testa di Y (Senza print!)
Y.head()

# %%
# Estrae un sottoinsieme di colonne passando una lista di nomi. Il risultato è un nuovo DataFrame.
X = iris[["sepal_length","sepal_width","petal_length","petal_width"]]
# Apparirà esattamente come nella prima tabella della tua immagine
X.head()

# %%
# Crea un nuovo DataFrame rimuovendo la colonna "species". 
# axis=1 indica a Pandas di cercare e rimuovere una colonna (mentre axis=0 agirebbe sulle righe).
X = iris.drop("species", axis=1)
# Apparirà esattamente come nella seconda tabella della tua immagine
X.head()

# %%
# Crea una copia fisica indipendente del DataFrame in memoria, utile per non modificare i dati originali per sbaglio.
iris_sampled = iris.copy()

# %%
# Estrae un campione casuale di righe dal DataFrame. 
# frac=1 significa che estrae il 100% delle righe, ottenendo l'effetto pratico di mescolare l'intero dataset.
iris_sampled = iris.sample(frac=1)
# Apparirà esattamente come nella terza tabella della tua immagine
iris_sampled.head()

# %%
# iloc - indice relativo 
# Estrae la primissima riga del DataFrame in base alla sua posizione posizionale (0 è la prima riga).
iris_sampled.iloc[0]

# %%
# loc - indice assoluto
# Estrae la riga che possiede esattamente l'etichetta di indice numero 134 (indipendentemente da dove si trovi fisicamente dopo il mescolamento).
iris_sampled.loc[134]

# %%
# Estrae un singolo valore all'incrocio tra l'etichetta di riga (32) e l'etichetta di colonna ("species").
iris_sampled.loc[32,"species"]

# %%
# Estrae le righe dalla posizione 0 alla 9 (esclusa la 10) per la colonna alla posizione 1 (la seconda colonna).
iris_sampled.iloc[:10,1]

# %%
# Ricarica il DataFrame indicando che la riga 0 del CSV contiene le vecchie intestazioni (header=0) da ignorare/sostituire con 'names'.
iris = pd.read_csv(
    iris_path, 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)

# %%
# Restituisce una tupla (numero_di_righe, numero_di_colonne) per capire le dimensioni del DataFrame.
iris.shape

# %%
# Genera un riassunto statistico (media, deviazione standard, min, max, quartili).
# include='all' forza il calcolo anche per le colonne non numeriche (come le stringhe in 'species').
iris.describe(include='all')

# %%
# Trova e restituisce il valore numerico più alto presente nella colonna 'sepal_length'.
iris['sepal_length'].max()

# %%
# Conta quanti valori non-nulli (non vuoti) esistono all'interno della colonna 'sepal_length'.
iris['sepal_length'].count()

# %%
# Calcola la varianza matematica (dispersione dei dati rispetto alla media) per la colonna.
iris['sepal_length'].var()
# %%
# Restituisce un array contenente tutti i valori unici presenti nella colonna 'species', rimuovendo i duplicati.
iris['species'].unique()


# CREAZIONE MASCHERA
# %%
iris = pd.read_csv(
    iris_path, 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
# Crea una maschera booleana (una sequenza di True e False). 
# Sarà True per le righe in cui 'petal_length' è maggiore della media globale di quella colonna, False altrimenti.
long_petal_mask = iris["petal_length"] > iris["petal_length"].mean()
# long_petal_mask

# Applica la maschera al DataFrame: mantiene e restituisce solo le righe che corrispondono al valore True.
iris_long_petals = iris[long_petal_mask]
iris_long_petals.head()


# %%
iris = pd.read_csv(
    iris_path, 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
iris_copy = iris.copy()
# Cerca tutte le righe in cui la colonna 'species' vale esattamente "Setosa", e sostituisce quel valore specifico con "undefined".
iris_copy.loc[ iris_copy["species"] == "Setosa" , "species"] = "undefined"
iris_copy["species"].unique()


# NORMALIZZAZIONE
# %%
iris = pd.read_csv(
    iris_path, 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
# escludiamo la colonna delle specie essendo non numerica
X = iris.drop("species",axis=1)
# sottraiamo il valore minimo e dividiamo per la differenza tra il valore massimo e il valore minimo
# (Questo processo ricalcola tutti i valori numerici scalandoli in un range tra 0 e 1)
X_norm = (X - X.min())/(X.max() - X.min())
X_norm.head()


# ORDINAMENTO
# %%
# Ordina l'intero DataFrame in base ai valori crescenti presenti nella colonna 'petal_length'.
iris.sort_values("petal_length").head

# RAGGRUPPAMENTO
# %%
# Divide l'intero DataFrame in gruppi separati, uno per ogni valore unico presente in 'species'.
grouped_species = iris.groupby("species")
# Calcola la media di tutte le colonne numeriche, ma lo fa separatamente per ogni gruppo creato in precedenza.
grouped_species.mean()
# I valori che vedi a schermo rappresentano la media matematica delle misurazioni, calcolata separatamente per ciascuna specie di fiore.


# %%
import numpy as np
# Applica la funzione 'np.count_nonzero' (che conta i valori diversi da zero) iterando su ogni singola riga (axis=1).
iris.apply(np.count_nonzero,axis=1).head()
# %%
# Applica la stessa funzione iterando però su ogni singola colonna dall'alto verso il basso (axis=0).
iris.apply(np.count_nonzero,axis=0).head()


# %%
iris = pd.read_csv(
    iris_path, 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
X = iris.drop('species',axis=1)
X.head()
# %%
# Utilizza map() per applicare la funzione lambda (arrotondamento e conversione a intero) ad ogni singola cella (valore) del DataFrame X.
X = X.map(lambda val:int(round(val,0)))
#X = X.round(0).astype(int)
X.head()


# %%
import pandas as pd
import numpy as np


iris = pd.read_csv(
    iris_path, 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)

iris_nan = iris.copy()
# Genera 10 numeri interi casuali, compresi tra 0 e il numero totale di righe del DataFrame (iris.shape[0]).
samples = np.random.randint(iris.shape[0],size=10)
# Usa gli indici casuali appena generati per localizzare quelle specifiche righe nella colonna 'petal_length' e inserisce il valore nullo (None/NaN).
iris_nan.loc[samples,'petal_length'] = None
# isnull() crea una mappa di True/False per i valori mancanti; sum() conta quanti True (valori mancanti) ci sono in quella colonna.
iris_nan['petal_length'].isnull().sum()
# %%
# Calcola la media della colonna, ignorando matematicamente in automatico le celle che contengono NaN.
mean_petal_length = iris_nan['petal_length'].mean()
#iris_nan['petal_length'].fillna(mean_petal_length, inplace=True)
# Sostituisce tutti i valori NaN presenti in 'petal_length' con il valore medio calcolato prima, e sovrascrive la colonna.
iris_nan['petal_length'] = iris_nan['petal_length'].fillna(mean_petal_length)
iris_nan['petal_length'].isnull().sum()

# %%
import matplotlib.pyplot as plt
# Genera un grafico a dispersione (scatter plot).
# Usa 'sepal_length' come asse X, 'sepal_width' come asse Y, tracciando un punto per ogni riga del DataFrame.
iris.plot(x='sepal_length', y='sepal_width', kind='scatter')
# %%

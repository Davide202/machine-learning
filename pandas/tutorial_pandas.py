# %%
import pandas as pd
import os

print(os.getcwd())


iris = pd.read_csv(
    "../data/iris.csv", 
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
Y = iris['species']
# Mostra la testa di Y (Senza print!)
Y.head()

# %%
X = iris[["sepal_length","sepal_width","petal_length","petal_width"]]
# Apparirà esattamente come nella prima tabella della tua immagine
X.head()

# %%
X = iris.drop("species", axis=1)
# Apparirà esattamente come nella seconda tabella della tua immagine
X.head()

# %%
iris_sampled = iris.copy()

# %%
iris_sampled = iris.sample(frac=1)
# Apparirà esattamente come nella terza tabella della tua immagine
iris_sampled.head()

# %%
# iloc - indice relativo 
iris_sampled.iloc[0]

# %%
# loc - indice assoluto
iris_sampled.loc[134]

# %%
iris_sampled.loc[32,"species"]

# %%
iris_sampled.iloc[:10,1]

# %%
iris = pd.read_csv(
    "../data/iris.csv", 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)

# %%
iris.shape

# %%
iris.describe(include='all')

# %%
iris['sepal_length'].max()

# %%
iris['sepal_length'].count()

# %%
iris['sepal_length'].var()
# %%
iris['species'].unique()


# CREAZIONE MASCHERA
# %%
iris = pd.read_csv(
    "../data/iris.csv", 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
long_petal_mask = iris["petal_length"] > iris["petal_length"].mean()
# long_petal_mask
iris_long_petals = iris[long_petal_mask]
iris_long_petals.head()


# %%
iris = pd.read_csv(
    "../data/iris.csv", 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
iris_copy = iris.copy()
iris_copy.loc[ iris_copy["species"] == "Setosa" , "species"] = "undefined"
iris_copy["species"].unique()


# NORMALIZZAZIONE
# %%
iris = pd.read_csv(
    "../data/iris.csv", 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
# escludiamo la colonna delle specie essendo non numerica
X = iris.drop("species",axis=1)
# sottraiamo il valore minimo e dividiamo per la differenza tra il valore massimo e il valore minimo
X_norm = (X - X.min())/(X.max() - X.min())
X_norm.head()


# ORDINAMENTO
# %%
iris.sort_values("petal_length").head

# RAGGRUPPAMENTO
# %%
grouped_species = iris.groupby("species")
grouped_species.mean()
# I valori che vedi a schermo rappresentano la media matematica delle misurazioni, calcolata separatamente per ciascuna specie di fiore.


# %%
import numpy as np
iris.apply(np.count_nonzero,axis=1).head()
# %%
iris.apply(np.count_nonzero,axis=0).head()


# %%
iris = pd.read_csv(
    "../data/iris.csv", 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)
X = iris.drop('species',axis=1)
X.head()
# %%
X = X.map(lambda val:int(round(val,0)))
#X = X.round(0).astype(int)
X.head()


# %%
import pandas as pd
import numpy as np


iris = pd.read_csv(
    "../data/iris.csv", 
    header=0, # =None,
    names=["sepal_length","sepal_width","petal_length","petal_width","species"]
)

iris_nan = iris.copy()
samples = np.random.randint(iris.shape[0],size=10)
iris_nan.loc[samples,'petal_length'] = None
iris_nan['petal_length'].isnull().sum()
# %%
mean_petal_length = iris_nan['petal_length'].mean()
#iris_nan['petal_length'].fillna(mean_petal_length, inplace=True)
iris_nan['petal_length'] = iris_nan['petal_length'].fillna(mean_petal_length)
iris_nan['petal_length'].isnull().sum()

# %%
import matplotlib.pyplot as plt
iris.plot(x='sepal_length', y='sepal_width', kind='scatter')
# %%

# TIPI DI VARIABILI:
# quantitativa continua
# qualitativa ordinata (categorical, ordinal) - associamo dei numeri
# qualitativa sconnessa (categorical, nominal) - one hot encoding (vero/falso per ogni proprietà : variabili di comodo)

# %%
## GESTIONE DATI MANCANTI
import pandas as pd
import numpy as np

iris_path = "../data/iris.csv"
iris_nan = pd.read_csv(
    iris_path,
    names=["sepal.length","sepal.width","petal.length","petal.width","variety"]
)

Y = iris_nan["variety"].values 
X = iris_nan.drop("variety",axis=1).values

iris_nan.head(10)

# %%
# rimuove le righe con dati mancanti
iris_drop = iris_nan.dropna()
iris_drop.head()

# %% 
# rimuove le colonne
iris_drop = iris_nan.dropna(axis=1)
iris_drop.head()

# %% 
# sostituzione dati mancanti con la media
# AGGIUNTO numeric_only=True per ignorare la colonna testuale "variety"
replace_with = iris_nan.mean(numeric_only=True)
iris_imp = iris_nan.fillna(replace_with)
iris_imp.head()

# %% 
# sostituzione dati mancanti con la mediana
# AGGIUNTO numeric_only=True
replace_with = iris_nan.median(numeric_only=True)
iris_imp = iris_nan.fillna(replace_with)
iris_imp.head()

# %% 
# sostituzione dati mancanti con la moda
# siccome la moda ritorna un dataframe degli elementi ordinati
# dobbiamo selezionare soltanto i primi valori
replace_with = iris_nan.mode().iloc[0]
iris_imp = iris_nan.fillna(replace_with)
iris_imp.head()

# %%
# AGGIORNAMENTO: Imputer è diventato SimpleImputer
from sklearn.impute import SimpleImputer

# axis=0 è stato rimosso, lavora già sulle colonne. missing_values usa np.nan
imp = SimpleImputer(strategy="mean", missing_values=np.nan)
# CORRETTO l'errore di battitura (trasform -> transform)
X_imp = imp.fit_transform(X)

# %%
imp = SimpleImputer(strategy="median", missing_values=np.nan)
X_imp = imp.fit_transform(X)

# %%
imp = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
X_imp = imp.fit_transform(X)
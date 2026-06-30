# TIPI DI VARIABILI:
# quantitativa continua
# qualitativa ordinata (categorical, ordinal) - associamo dei numeri
# qualitativa sconnessa (categorical, nominal) - one hot encoding (vero/falso per ogni proprietà : variabili di comodo)


# %%
## GESTIONE DATI MANCANTI
import pandas as pd
import numpy as np

iris_path = "./data/iris.csv"
iris_nan = pd.read_csv(iris_path)

Y = iris_nan["class"].values 
X = iris_nan.drop("class",axis=1).values
X.head()

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
replace_with = iris_nan.mean()
iris_imp = iris_nan.fillna(replace_with)
iris_imp.head()

# %% 
# sostituzione dati mancanti con la mediana
replace_with = iris_nan.median()
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
from sklearn.preprocessing import Imputer

imp = Imputer(strategy="mean", axis=0, missing_values="NaN")
X_imp = imp.fit_trasform(X)


# %%
imp = Imputer(strategy="median", axis=0, missing_values="NaN")
X_imp = imp.fit_trasform(X)
# %%
imp = Imputer(strategy="most_frequent", axis=0, missing_values="NaN")
X_imp = imp.fit_trasform(X)
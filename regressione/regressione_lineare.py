# Apprendimento Supervisionato - Regressione Lineare
# La regressione lineare consiste nel tracciare la retta
# più vicina possibile a tutti i punti del grafico (x,y)

# Equazione della retta in forma esplicita y = mx + q
# y = variabile dipendente
# q = intercetta
# m = coefficiente angolare
# x = variabile indipendente

# q e m sono i parametri che la la regressione deve utilizzare
# per trovare la retta che meglio approssima la relazione tra x e y dei punti dati.

## FUNZIONE COSTO fornisce una misura di quanto la retta stimata si discosta dai punti dati.
# esistono vari tipi di funzioni costo:

# errore quadratico medio (MSE)
# errore assoluto medio (MAE)

# somma dei quadrati degli errori (somma dei quadrati spiegati) (SSE)
# somma dei quadrati degli errori normalizzati (SSEN)

# somma dei quadrati residuali (SSR)
# somma dei quadrati residui (RSS) è definita come la somma degli errori al quadrato per ogni punto del dataset. 
# per errore si intende la differenza tra il valore osservato (corretto) e il valore predetto dalla retta stimata.

# somma dei quadrati totali (SST)

# somma dei valori assoluti degli errori (SAE)
# somma dei valori assoluti degli errori normalizzati (SANE)

# Minore è il valore della funzione costo, migliore è la stima della retta,
# maggiore è la qualità del nostro modello predittivo.
# I valori dei pesi ideali sono quelli che minimizzano la funzione costo.

## Il GRADIENT DESCEND è un algoritmo di ottimizzazione iterativo 
# che permette di trovare i valori dei pesi (q e m) che minimizzano la funzione costo.

# %%
import pandas as pd
import numpy as np

from dataset.splitting_dataset import X_train

housing_path = "../data/housing.csv"
dataframe = pd.read_csv(housing_path)

boston = dataframe.copy()
boston.columns = ["RM","MEDV"]
boston.head()

# %%

X = dataframe["RM"].values
Y = dataframe["MEDV"].values

X.head()
Y.head()

# %%

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.linear_model import LinearRegression

ll = LinearRegression()

ll.fit(X_train,Y_train)

Y_pred = ll.predict(X_test)

# %% 

from sklearn.metrics import mean_squared_error , r2_score

mean_squared_error(Y_test,Y_pred) 

r2_score(Y_test,Y_pred)

# %%

import matplotlib.pyplot as plt

print("Peso di RM: " + str(ll.coef_[0]))
print("Bias: " + str(ll.intercept_))

plt.scatter(X_train, Y_train, color='green',edgecolor="white", label='Train Set')
plt.scatter(X_test, Y_test, color='blue', edgecolor="white", label='Test Set')


plt.xlabel("Numero medio di stanze [RM]")
plt.ylabel("Valore in $1000 [MEDV]")

plt.legend(loc="upper left")

plt.plot(X_test, Y_pred, color='red', linewidth=3)
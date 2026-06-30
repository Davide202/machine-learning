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
#import numpy as np

housing_path = "../data/housing.csv"
dataframe = pd.read_csv(housing_path)

# %%
# Estraiamo SOLO le due colonne che ci interessano
boston = dataframe[["rm", "medv"]].copy()
boston.head()

# %%
# DOPPIE PARENTESI per X (per avere una matrice 2D per Scikit-Learn)
X = dataframe[["rm"]].values
# SINGOLA PARENTESI per Y (per avere una lista 1D)
Y = dataframe["medv"].values

# Usiamo lo slicing di Python per vedere i primi 5 valori dell'array
print("Primi 5 valori di X:\n", X[:5])
print("Primi 5 valori di Y:\n", Y[:5])

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dividiamo i dati
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Creiamo il modello e lo addestriamo
ll = LinearRegression()
ll.fit(X_train, Y_train)

# Facciamo le previsioni
Y_pred = ll.predict(X_test)

# %% 
from sklearn.metrics import mean_squared_error, r2_score

# Stampiamo i risultati delle metriche per vederli a schermo
print("Mean Squared Error (MSE):", mean_squared_error(Y_test, Y_pred)) 
print("R2 Score:", r2_score(Y_test, Y_pred))

# %%
import matplotlib.pyplot as plt

print("Peso di RM (Coefficiente): " + str(ll.coef_[0]))
print("Bias (Intercetta): " + str(ll.intercept_))

# Disegniamo i punti di addestramento e di test
plt.scatter(X_train, Y_train, color='green', edgecolor="white", label='Train Set')
plt.scatter(X_test, Y_test, color='blue', edgecolor="white", label='Test Set')

# Impostiamo le etichette per gli assi
plt.xlabel("Numero medio di stanze [RM]")
plt.ylabel("Valore in $1000 [MEDV]")
plt.legend(loc="upper left")

# Disegniamo la retta di regressione!
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.show() # Consigliato per forzare la stampa del grafico
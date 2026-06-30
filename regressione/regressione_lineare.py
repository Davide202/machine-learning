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

housing_path = "../data/housing.csv"
boston = pd.read_csv(housing_path)












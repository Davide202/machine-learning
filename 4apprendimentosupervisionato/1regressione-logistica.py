# Classificazione : Regressione Logistica


# %% 
# Dataset: 

import pandas as pd

file_path = "../data/cancer/wdbc.data"

# Le 10 caratteristiche di base estratte dal file wdbc.names
base_features = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness', 
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]

# Generiamo automaticamente le 30 colonne aggiungendo i suffissi (mean, se, worst)
mean_features = [f + '_mean' for f in base_features]
se_features = [f + '_se' for f in base_features]
worst_features = [f + '_worst' for f in base_features]

# Uniamo l'ID, la diagnosi e tutte le 30 misurazioni nell'ordine corretto
headers = ['id', 'diagnosis'] + mean_features + se_features + worst_features



# Carica il DataFrame indicando che non c'è una riga di intestazione
df = pd.read_csv(file_path, header=None, names=headers)
df.head()

# (Opzionale ma consigliato) Separiamo subito X (dati) e y (diagnosi)
# La colonna 0 è l'ID, non ci serve per il machine learning.
# La colonna 1 è la diagnosi (M o B).
# Dalla colonna 2 fino alla fine (31) ci sono i 30 valori numerici.

X = df.iloc[:, 2:] # Prendi tutte le righe, dalla terza colonna in poi
Y = df.iloc[:, 1]  # Prendi tutte le righe, ma solo la seconda colonna

# Mostra a schermo i dati
print(X.head())
print(Y.head())

# %%

# Selezioniamo solo le colonne che ci interessano passandole come lista
colonne_da_estrarre = ['radius_mean', 'concave_points_mean', 'diagnosis']

# Creiamo un nuovo DataFrame "ridotto"
df_ridotto = df[colonne_da_estrarre]

# Visualizziamo le prime righe per confermare il risultato
print(df_ridotto.head(10))

# Classificazione binomiale: Maligno (M) -> 1 , Benigno (B) -> 0

# Lo scopo di un algoritmo di classificazione è quello di trovare una retta che meglio separa i dati in due classi. 
# In questo caso, vogliamo separare i tumori maligni da quelli benigni.   



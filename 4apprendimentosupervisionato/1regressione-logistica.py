# Classificazione : Regressione Logistica


# Classificazione binomiale: Maligno (M) -> 1 , Benigno (B) -> 0

# Lo scopo di un algoritmo di classificazione è quello di trovare una retta che meglio separa i dati in due classi. 
# In questo caso, vogliamo separare i tumori maligni da quelli benigni.   

# FUNZIONE DI ATTIVAZIONE

# La funzione di classificazione mappa i punti dati in un valore reale z,
# positivo se il punto si trova al di sopra della retta di separazione e negativo se si trova al di sotto.

# La funzione di attivazione (sigmoid) mappa qualsiasi valore reale in un intervallo compreso tra 0 e 1,
# rendendola adatta per problemi di classificazione binaria. 
# o.5 è la soglia di attivazione: 
#   se il valore di output è maggiore di 0.5, il modello classifica il punto come appartenente alla classe positiva (1), 
#   altrimenti lo classifica come appartenente alla classe negativa (0). 
# Chiameremo tale funzione fi(z) = 1 / (1 + e^(-z)), dove e è la base dei logaritmi naturali.


# La Regressione Logistica è un modello di classificazione 
# che utilizza come funzione di attivazione la funzione logistica (sigmoidale).    

# La regressione logistica si addestra nello stesso modo in cui si addestra un modello di regressione,
# la regressione logistica non è altro che una regressione lineare a cui si applica un caso particolare 
# di funzione logistica (sigmoidale) come funzione di attivazione per eseguire la classificazione.

# L'unica cosa che cambia è la funzione di costo. 
# Che in questo caso più che di costo si parla di affinità tra la distribuzione dei dati e la distribuzione predetta dal modello.
# La più utilizzata è la Likelihood Function (funzione di massima verosimiglianza), 
# che misura quanto bene il modello si adatta ai dati osservati.  
# Date le proprietà X ed i pesi W qual'è la probabilità di ottenere l'output desiderato Y?

# Per una funzione numerica, durante la fase di addestramento conviene utilizzare la Log Likelihood,
# che corrisponde al logaritmo della Likelihood Function.
# Durante l'addestramento bisogna massimizzare la Log Likelihood Function, 
# tramite un algoritmo di ottimizzazione chiamato Gradient Ascend, 
# che massimizza la funzione di costo.

# %% 
# Dataset: 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss 

file_path = "../data/cancer/wdbc.data"

headers = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Carica il DataFrame indicando che non c'è una riga di intestazione
breast_cancer = pd.read_csv(file_path, header=None, names=headers)

#breast_cancer.head()

breast_cancer.info()

# vediamo quali sono le classi da classificare
breast_cancer['diagnosis'].unique()

# %%
# Iniziamo eseguendo una classificazione utilizzando soltanto due proprietà:
# radius_se (errore standard del raggio del tumore) 
# concave_points_worst (numero peggiore di punti di concavità nel contorno del tumore)

X = breast_cancer[['radius_se', 'concave_points_worst']].values
Y = breast_cancer['diagnosis'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)    

# Utilizziamo il LabelEncoder per convertire le etichette di classe in valori numerici (0 e 1)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

Y_train[:5]


# %% 
# Standardizziamo il dataset per avere media 0 e deviazione standard 1,
# in modo che le caratteristiche abbiano la stessa scala e il modello possa convergere più velocemente durante l'addestramento.     

ss = StandardScaler()
X_train = ss.fit_transform(X_train)     
X_test = ss.transform(X_test)

# %%
# Eseguiamo la regressione logistica utilizzando la classe LogisticRegression di scikit-learn.

#lr = LogisticRegression(random_state=0)
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# Adesso che abbiamo il nostro modello addestrato possiamo testarlo sui dati di test e vedere come si comporta.
# Eseguiamo le predizioni sui dati di test e calcoliamo l'accuratezza del modello.

Y_pred = lr.predict(X_test)
#Y_pred_proba = lr.predict_proba(X_test) [:, 1]  # Probabilità della classe positiva
Y_pred_proba = lr.predict_proba(X_test)

print("Accuracy: " + str(accuracy_score(Y_test, Y_pred)))
print("Log Loss: " + str(log_loss(Y_test, Y_pred_proba)))


# %%
# Proviamo a visualizzare il Decision Boundary del modello addestrato. 

from viz import showBounds 
showBounds(lr, X_train, Y_train, labels=['Benigno', 'Maligno'], title='Decision Boundary - Logistic Regression')

# %%

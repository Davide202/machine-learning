
# %%


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

wines_path = "../data/wine/wine.csv"

cols = [
    "label",
    "alcol",
    "acido malico",
    "cenere",
    "alcalinità della cenere",
    "magnesio",
    "fenoli totali",
    "flavonoidi",
    "fenoli non-flavonoidi",
    "proantocianidine",
    "intensità del colore",
    "tonalità",
    "OD280/OD315 dei vini diluiti",
    "proalina"
]

wines = pd.read_csv(wines_path,
                    #usecols=[0,1,7],
                    header=0,
                    names=cols
                    )
wines.head(10)


Y = wines['label'].values
X = wines.drop('label',axis=1).values

# applichiamo la standardizzazione

ss = StandardScaler()
X = ss.fit_transform(X)

from sklearn.decomposition import PCA 

pca = PCA(n_components=2)

X_pc = pca.fit_transform(X)

X_pc

plt.figure(figsize=[14,10])
plt.xlabel("Prima Componente Principale")
plt.xlabel("Seconda Componente Principale")
plt.scatter(X_pc[:,0],X_pc[:,1], c=Y, edgecolor='black',cmap='viridis', s=60)
plt.title("Analisi delle Componenti Principali (PCA) sul Dataset Wine")
plt.show()

# dal grafico si nota che: 
# anche se abbiamo ridotto la dimensionalità del dataset a 2
# le classi continuano ad essere ben definite


# Supponiamo di voler mantenere la maggiore quantità possibile di informazione e quindi varianza.
# Per avere un idea del numero di componenti principali da tenere possiamo visualizzare graficamente 
# la percentuale di varianza contenuta in ogni componente principale insieme alla varianza cumulativa  
# di tutte le possibili componenti. 

pca = PCA(n_components=None)

pca.fit(X)

print("VARIANZE COMPONENTI PRINCIPALI: " + str(pca.explained_variance_ratio_))


plt.figure(figsize=[14,10])
plt.step(range(1,14),np.cumsum(pca.explained_variance_ratio_),where='mid')
plt.bar(range(1,14),pca.explained_variance_ratio_)
plt.xlabel("Varianza")
plt.xlabel("Componenti principali")
plt.title("Varianza delle componenti principali")
plt.show()

# Notiamo che considerando le prime 5 componenti principali
# arriviamo all'80% della varianza totale.

import sys
# Aggiungi il percorso della cartella che contiene leggimnist.py. 
# Ad esempio, se si trova nella cartella di livello superiore usa '../'
sys.path.append('./data')
sys.path.append('../data')

from leggimnist import carica_immagini_mnist, carica_etichette_mnist

X_train = carica_immagini_mnist('train-images.idx3-ubyte')
Y_train = carica_etichette_mnist('train-labels.idx1-ubyte')

X_test = carica_immagini_mnist('t10k-images.idx3-ubyte')
Y_test = carica_etichette_mnist('t10k-labels.idx1-ubyte')

# Stampiamo le dimensioni per confermare il successo dell'estrazione
print(f"Dimensioni del Training Set (Immagini): {X_train.shape}")  # Atteso: (60000, 784)
print(f"Dimensioni del Training Set (Etichette): {Y_train.shape}") # Atteso: (60000,)

print(f"Dimensioni del Test Set (Immagini): {X_test.shape}")       # Atteso: (10000, 784)
print(f"Dimensioni del Test Set (Etichette): {Y_test.shape}")      # Atteso: (10000,)


from matplotlib.pyplot import imshow
imshow(X_test[0].reshape([28,28]),cmap='gray')

Y_test[0]

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

from time import time

lr = LogisticRegression()
start_time = time()
lr.fit(X_train,Y_train)
end_time = time()

print("TEMPO DI ADDESTRAMENTO: " + str(end_time - start_time))

accuracy_train = accuracy_score(Y_train, lr.predict(X_train))
accuracy_test = accuracy_score(Y_test, lr.predict(X_test))

loss_train = log_loss(Y_train, lr.predict_proba(X_train))
loss_test = log_loss(Y_test, lr.predict_proba(X_test))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))




# %%
# Utilizziamo la PCA per ridurre il tempo di addestramento senza distruggere le metriche del modello.


pca = PCA(0.95) # specifichiamo che vogliamo mantenere il 95% della varianza 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca.shape
X_test_pca.shape  

lr = LogisticRegression()
start_time = time()
lr.fit(X_train_pca,Y_train)
end_time = time()

print("TEMPO DI ADDESTRAMENTO: " + str(end_time - start_time))

accuracy_train = accuracy_score(Y_train, lr.predict(X_train_pca))
accuracy_test = accuracy_score(Y_test, lr.predict(X_test_pca))

loss_train = log_loss(Y_train, lr.predict_proba(X_train_pca))
loss_test = log_loss(Y_test, lr.predict_proba(X_test_pca))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))



# %%
# Riduciamo la percentuale di varianza


pca = PCA(0.9) # specifichiamo che vogliamo mantenere il 95% della varianza 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca.shape
X_test_pca.shape  

lr = LogisticRegression()
start_time = time()
lr.fit(X_train_pca,Y_train)
end_time = time()

print("TEMPO DI ADDESTRAMENTO: " + str(end_time - start_time))

accuracy_train = accuracy_score(Y_train, lr.predict(X_train_pca))
accuracy_test = accuracy_score(Y_test, lr.predict(X_test_pca))

loss_train = log_loss(Y_train, lr.predict_proba(X_train_pca))
loss_test = log_loss(Y_test, lr.predict_proba(X_test_pca))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))

# %%
# Utilizziamo LogisticRegression(solver='lbfgs')


pca = PCA(0.9) # specifichiamo che vogliamo mantenere il 95% della varianza 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca.shape
X_test_pca.shape  

lr = LogisticRegression(solver='lbfgs')
start_time = time()
lr.fit(X_train_pca,Y_train)
end_time = time()

print("TEMPO DI ADDESTRAMENTO: " + str(end_time - start_time))

accuracy_train = accuracy_score(Y_train, lr.predict(X_train_pca))
accuracy_test = accuracy_score(Y_test, lr.predict(X_test_pca))

loss_train = log_loss(Y_train, lr.predict_proba(X_train_pca))
loss_test = log_loss(Y_test, lr.predict_proba(X_test_pca))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))


# %%

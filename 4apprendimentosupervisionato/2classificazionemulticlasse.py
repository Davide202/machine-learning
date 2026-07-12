# Riconoscimento di immagini con classificazione multiclasse
# In questo esempio, utilizzeremo il dataset MNIST, che contiene immagini di cifre scritte a mano (0-9). 
# L'obiettivo è costruire un modello di classificazione multiclasse per riconoscere le cifre.    






# %%

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# # Scarica il dataset MNIST originale (ci metterà qualche secondo)
# print("Scaricamento di MNIST in corso...")
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# # Separiamo le immagini (X) dalle etichette (y)
# X = mnist.data
# y = mnist.target

# print(f"Dimensioni di X: {X.shape}") # (70000, 784) -> 70.000 immagini da 28x28 pixel
# print(f"Dimensioni di y: {y.shape}") # (70000,) -> 70.000 etichette

# # Prendiamo la prima immagine (riga 0) e la rimodelliamo in una griglia 28x28
# prima_immagine = X[0].reshape(28, 28)
# etichetta = y[0]

# plt.imshow(prima_immagine, cmap='gray')
# plt.title(f"Etichetta reale: {etichetta}")
# plt.axis('off') # Nasconde gli assi
# plt.show()


# Carica il dataset delle cifre (8x8 pixel)
digits = load_digits()  

X = digits.data
Y = digits.target

X.shape #(1797, 64) -> 1797 immagini da 8x8 pixel
Y.shape #(1797,) -> 1797 etichette  

np.unique(Y) 

for i in range(10):
    pic_matrix = X[Y == i][0].reshape([8, 8])  # Prende la prima immagine per ogni cifra
    plt.subplot(2, 5, i + 1)  # Crea una griglia di 2 righe e 5 colonne
    plt.imshow(pic_matrix, cmap='gray')  # Mostra l'immagine        
    plt.title(f"Etichetta: {Y[i]}")  # Mostra l'etichetta dell'immagine
    plt.axis('off')  # Nasconde gli assi  
    plt.show()  


# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# normalizziamo tutte le colonne di X_train e X_test in modo che abbiano media 0 e deviazione standard 1

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

print("Accuracy: " + str(accuracy_score(Y_test, Y_pred)) )
print("Log Loss: " + str(log_loss(Y_test, Y_pred_proba)) )

# %%
# Introduciamo una nuova metriva di valutazione: la matrice di confusione.
# Ci permette di vedere su quali classi il modello fa errori di classificazione 
# e quali classi vengono predette correttamente.  

from sklearn.metrics import confusion_matrix    

cm = confusion_matrix(Y_test, Y_pred)
print("Matrice di Confusione:")
print(cm)   

# %%

import seaborn as sns

plt.figure(figsize=(10, 10))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, square=True)
plt.xlabel('Classe predetta') 
plt.ylabel('Classe corretta')   
plt.title('Matrice di Confusione')
plt.show()

# %%

from sklearn.multiclass import OneVsRestClassifier 

ovr = OneVsRestClassifier(LogisticRegression())

ovr.fit(X_train,Y_train)

Y_pred = ovr.predict(X_test)
Y_pred_proba = ovr.predict_proba(X_test)

print("Accuracy: " + str(accuracy_score(Y_test, Y_pred)) )
print("Log Loss: " + str(log_loss(Y_test, Y_pred_proba)) )

# %%

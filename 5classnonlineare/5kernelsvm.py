# KERNEL SVM
# Vediamo come utilizzare SVM 
# (Support Vector Machines o Macchine a Vettore di Supporto)
# per eseguire classificazioni non lineari utilizzando il così detto Kernel Trick.

# Consideriamo i casi non è possibile costruire una retta in grado di separare due classi,
# quindi non è possibile trovare una retta da utilizzare come classificatore. 

# In questo caso è possibile sfruttare la strategia di mappare i dati in una dimensione maggiore,
# aggiungendo una proprietà di supporto, al fine di ottenere la possibilità di separare le classi
# con un iperpiano. Successivamente riproiettando i punti e l'iperpiano nello spazio di partenza,
# otterremo che la proiezione dell'iperpiano è una curva che separa le classi,
# cioè un classificatore non lineare  in grado di separare le classi in modo efficiente.

# Problema: Mappare i dati in una dimensione superiore, 
# costruendo una nuova proprietà, è estremamente dispendioso in termini
# di risorse di calcolo! 

# Per risolvere questo problema entra in gioco il Kernel Trick
# in grado di simulare lo stesso effetto di una proiezione in una dimensione maggiore
# utilizzando una funzione kernel.

# Una funzione kernel è sostanzialmente una metrica che permette di misurare
# la somiglianza tra due esempi.

# Esistono diverse funzioni kernel, la più utilizzata è il kernel Gaussiano (Radial Basis),
# tale funzione assume valori tra 0 per esempi molto diversi e 1 per esempi molto simili,
# la costante sigma rappresenta la sensibilità alla differenza tra gli esempi.

# Prediamo un punto chiamato Landmark e posizioniamolo esattamente al centro del grafico degli esempi,
# calcoliamo quindi il valore della funzione kernel per ciascun punto rispetto al landmark,
# il valore di questa funzione tenderà a 0 per gli esempi di una classe e ad 1 per gli esempi dell'altra. 

# Utilizzando questo stesso principio è possibile risolvere anche problemi più complessi,
# utilizzando diversi landmarks e calcolando il valore della funzione kernel per ogni esempio
# per ognuno dei landmark, possiamo utilizzare i valori così ottenuti come proprietà del nostro modello.

# Altre funzioni kernel:
# Kernel Lineare, che corrisponde ad una semplice applicazione delle svm e permette di ottenere
#   decision boundary rigorosamente lineari (prodotto scalare);
# Kernel Gaussiano o Radial Basis, questo è il kernel generico se non si sa come muoversi;
# Kernel Sigmoidale
# Kernel Polinomiale 
#   sebbene questi ultimi due nella pratica non abbiano gli stessi risultati del Gaussiano,
#   vale la pena provarli durante la fase di tuning degli iperparametri,
#   nel caso in cui i risultati dati dal Gaussiano non siano soddisfacenti.


# %% 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles 
from sklearn.svm import SVC
from viz_svm import plot_svm_bounds


# Creiamo il nostro dataset
X,Y = make_circles(noise=0.2, factor=0.5, random_state=1)
# il factor a 0.5 fa in modo che gli esempi positivi siano distribuiti verso il centro del grafico
# il random_state pari ad 1 permette di avere sempre lo stesso dataset
X.shape # (100, 2) otteniamo un dataset con 100 esempi distrubuiti in due proprietà
# Costruiamo uno scatterplot per visualizzare la distribuzione degli esempi su un grafico 2D
plt.scatter(X[:,0],X[:,1],c=Y)
# Vediamo se riusciamo ad eseguire una classificazione
# Per prima cosa costruiamo gli array per addestramento e test
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# %%
# Costruiamo la kernel svm ovvero la macchina a vettore di supporto
# che utilizza una funzione kernel per proiettare i dati in uno spazio di dimensione maggiore
# nel quale possiamo provare ad eseguire la classificazione.

# Come prima prova utilizziamo un Kernel lineare
# svc = SVC(kernel='linear', probability=True) il parametro probability è deprecato
svc = SVC(kernel='linear')
#cccv = CalibratedClassifierCV(svc, ensemble=False)
# il parametro  probability=True fa in modo che la macchina virtuale di supporto
# calcoli anche la probabilità di appartenenza ad una classe
svc.fit(X_train,Y_train)
accuracy_train = svc.score(X_train,Y_train)
accuracy_test = svc.score(X_test,Y_test)
print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
plot_svm_bounds(svc, X_train, Y_train, title="Fallimento del Kernel Lineare")


# %%
# Adesso proviamo con un altro Kernel visto che quello lineare ha fallito,
# Proviamo con il Kernel Gaussiano chiamato anche RBF
svc = SVC(kernel='rbf')
svc.fit(X_train,Y_train)
accuracy_train = svc.score(X_train,Y_train)
accuracy_test = svc.score(X_test,Y_test)
print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
from viz_svm import plot_svm_bounds
plot_svm_bounds(svc, X_train, Y_train, title="Kernel Gaussiano solo train")
plot_svm_bounds(svc, X_train, Y_train,X_test,Y_test, title="Kernel Gaussiano train e test")



# %%
# Testiamo il Kernel Sigmoidale 
svc = SVC(kernel='sigmoid')
svc.fit(X_train,Y_train)
accuracy_train = svc.score(X_train,Y_train)
accuracy_test = svc.score(X_test,Y_test)
print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
from viz_svm import plot_svm_bounds
plot_svm_bounds(svc, X_train, Y_train, title="Kernel Sigmoidale solo train")
plot_svm_bounds(svc, X_train, Y_train,X_test,Y_test, title="Kernel Sigmoidale train e test")


# %%
# Testiamo il Kernel Polinomiale 
svc = SVC(kernel='poly')
svc.fit(X_train,Y_train)
accuracy_train = svc.score(X_train,Y_train)
accuracy_test = svc.score(X_test,Y_test)
print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
from viz_svm import plot_svm_bounds
plot_svm_bounds(svc, X_train, Y_train, title="Kernel Polinomiale solo train")
plot_svm_bounds(svc, X_train, Y_train,X_test,Y_test, title="Kernel Polinomiale train e test")


# %%
# CONCLUSIONE: con questo dataset il Kernel Gaussiano fa da sovrano,
# però è anche possibile che con altre tipologie di dataset
# il kernel sigmoidale o il polinomiale possano rendere meglio,
#  quindi vale la pena aggiungere tra gli iperparametri da ottimizzare 
# durante la fase di tuning.
# Nearest Neighbor esegue una classificazione ritornando la classe dell'esempio più simile.
# Modello Lazy, non apprende dai dati ma li memorizza.

# Per poter confrontare gli esempi ci serve una metrica per poter calcolare la distanza.
# Abbiamo diverse opzioni per questo:
# Metrica Euclidea
# Manhattan
# Minkowski (p)

# K-Nearest Neighbors è simile al Nearest Neighbor,
# cerca i k esempi più vicini, ritorna la classe più presente in questo insieme,
# K controlla la complessità, un buon valore è 5.

# Esempio con k = 5 e metrica Euclidea

# Pro: estremamente semplice e si adatta subito a nuovi dati
# Contro: dispendioso per dataset grandi, bisogna sceglie la metrica corretta,
    # non esiste una regola universale per farlo.


# %% 

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()

X = digits.data 
Y = digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

# Normalizzazione
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# Adesso possiamo costruire il modello
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)

Y_pred_train = knn.predict(X_train)
Y_prob_train = knn.predict_proba(X_train)

Y_pred = knn.predict(X_test)
Y_prob = knn.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train,Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_pred)

loss_train = log_loss(Y_train,Y_prob_train)
loss_test = log_loss(Y_test,Y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f " % (accuracy_train,accuracy_test))

print("LOC_LOSS: TRAIN=%.4f TEST=%.4f " % (loss_train,loss_test))


# %% 
# Vediamo cosa succede se facciamo variare il valore di K
# Creiamo diversi modelli per diversi valori di k

Ks = [1,2,3,4,5,7,10,12,15,20]
for K in Ks:
    print("K=" + str(K))

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,Y_train)

    Y_pred_train = knn.predict(X_train)
    Y_prob_train = knn.predict_proba(X_train)

    Y_pred = knn.predict(X_test)
    Y_prob = knn.predict_proba(X_test)

    accuracy_train = accuracy_score(Y_train,Y_pred_train)
    accuracy_test = accuracy_score(Y_test,Y_pred)

    loss_train = log_loss(Y_train,Y_prob_train)
    loss_test = log_loss(Y_test,Y_prob)

    print("ACCURACY: TRAIN=%.4f TEST=%.4f " % (accuracy_train,accuracy_test))
    print("LOC_LOSS: TRAIN=%.4f TEST=%.4f " % (loss_train,loss_test))

# %%
# Prendiamo il nostro modello per K=3 
# e facciamo disegnare le classi che ha sbagliato a classificare.

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

Y_pred_train = knn.predict(X_train)
Y_prob_train = knn.predict_proba(X_train)

Y_pred = knn.predict(X_test)
Y_prob = knn.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train,Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_pred)

loss_train = log_loss(Y_train,Y_prob_train)
loss_test = log_loss(Y_test,Y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f " % (accuracy_train,accuracy_test))
print("LOC_LOSS: TRAIN=%.4f TEST=%.4f " % (loss_train,loss_test))

import matplotlib.pyplot as plt 

for i in range(0, len(X_test)):
    if(Y_pred[i] != Y_test[i]):
        print("Numero %d classificato come %d " % (Y_test[i],Y_pred[i]))
        plt.imshow(X_test[i].reshape([8,8]), cmap="gray")
        plt.show()
        






# %%

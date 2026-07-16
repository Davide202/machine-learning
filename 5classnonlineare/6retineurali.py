
# Deep Learning : insieme di metodi per l'addestramento di una rete neurale artificiale profonda.

# Vediamo come addestrare un modello di reti neurali.

# Implementeremo un PERCETTRONE MULTISTRATO per classificare un dataset di immagini di cifre scritte a mano.
# Una rete neurale artificiale ha bisogno di un numero molto elevato di esempi per poter essere addestrata correttamente,
# per questo il piccolo dataset di immagini delle cifre che ci mette a disposizione sklearn non è assolutamente sufficiente. 
# Quindi per il nostro scopo utilizzeremo il MNIST.

# %%
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

# %%
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss , accuracy_score 
from sklearn.preprocessing import MinMaxScaler 

# Normalizziamo il Dataset

mms = MinMaxScaler()

X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# Prima di utilizzare una rete neurale bisogna essere sicuri della sua necessità
# essendo un modello molto potente e complesso che necessita di tuning e ottimizzazioni
# perchè ci sono moltissimi parametri.

# Prima di passare all'implementazione del Percettone Multistrato 
# proviamo ad eseguire una semplice Regressione Logistica.

from sklearn.linear_model import LogisticRegression

# verbose=1 stampa l'avanzamento nel terminale così vedi che non è bloccato
# n_jobs=-1 dice a Python di usare TUTTI i core del tuo processore per fare prima
# solver='saga' è un algoritmo di calcolo ottimizzato per dataset giganteschi come MNIST
# Il risolutore saga gestisce matrici enormi molto meglio rispetto a quello standard (lbfgs).
lr = LogisticRegression(max_iter=1000, n_jobs=-1, verbose=1, solver='saga')

print("Inizio l'addestramento. Potrebbe volerci qualche minuto, guarda i log qui sotto...")
lr.fit(X_train,Y_train)
print("Addestramento completato!")

Y_pred_train = lr.predict(X_train)
Y_prob_train = lr.predict_proba(X_train)

Y_pred = lr.predict(X_test)
Y_prob = lr.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train,Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_test) 

loss_train = log_loss(Y_train,Y_prob_train)
loss_test = log_loss(Y_test,Y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test)) # 0.9279   0.9200
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test)) # 0.3138   0.3443 QUINDI QUESTO MODELLO NON VA BENE

# convergence after 512 epochs took 579 seconds
# Addestramento completato!
# ACCURACY: TRAIN=0.9394 TEST=1.0000
# LOG LOSS: TRAIN=0.2208 TEST=0.2713


# Questa lunga attesa è esattamente il motivo per cui, per dati complessi come le immagini, 
# la Regressione Logistica viene abbandonata in favore delle Reti Neurali.


# %%
# Vediamo come si comporta una Rete Neurale

from sklearn.neural_network import MLPClassifier 

mlp = MLPClassifier(hidden_layer_sizes=(100,) , verbose=True)

print("Inizio l'addestramento. Potrebbe volerci qualche minuto, guarda i log qui sotto...")
mlp.fit(X_train, Y_train)
print("Addestramento completato!")


Y_pred_train = mlp.predict(X_train)
Y_prob_train = mlp.predict_proba(X_train)

Y_pred = mlp.predict(X_test)
Y_prob = mlp.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train,Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_test) 

loss_train = log_loss(Y_train,Y_prob_train)
loss_test = log_loss(Y_test,Y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test)) # 0.9999   0.9766
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test)) # 0.022    0.0951

# %%
# Vediamo se possiamo fare di meglio
# Proviamo ad implementare una rete neurale profonda
# Ovvero una rete neurale con più di un hidden layers


mlp = MLPClassifier(hidden_layer_sizes=(512,512,) , verbose=True)

print("Inizio l'addestramento. Potrebbe volerci qualche minuto, guarda i log qui sotto...")
mlp.fit(X_train, Y_train)
print("Addestramento completato!")


Y_pred_train = mlp.predict(X_train)
Y_prob_train = mlp.predict_proba(X_train)

Y_pred = mlp.predict(X_test)
Y_prob = mlp.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train,Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_test) 

loss_train = log_loss(Y_train,Y_prob_train)
loss_test = log_loss(Y_test,Y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test)) # 0.99982   0.9797
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test)) # 0.0063    0.0919

# Le metriche ci dicono che il modello è migliorato seppur non di moltissimo
# a questi livelli, con accuracy quasi ad 1 ogni centesimo è già un guadagno.



# %% 
# Per concludere andiamo a vedere quali esempi la nostra rete neurale profonda
# ha fallito a classificare


for i in range(len(X_test)):
    if(Y_test[i] != Y_pred[i]):
        print("Numero %d classificato come %d" % (Y_test[i],Y_pred[i]) )
        plt.imshow(X_test[i].reshape([28,28]), cmap="gray")
        plt.show()

# Osservazione: guardando gli esempi sbagliati, 
# alcuni non sono riconoscibili nemmeno all'occhio umano
# quindi la nostra rete neurale profonda ha fatto un ottimo lavoro.
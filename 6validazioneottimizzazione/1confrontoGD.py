# Confronto dei vari tipi di classificatori Gradient Descend
# e delle loro funzioni di costo.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
from sklearn.utils import shuffle


X , Y = make_classification(n_samples=1250, n_features=4, n_informative=2, random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Per eseguire lo Stochastic Gradint Descend abbiamo a disposizione la classe:
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss="log")

# Possiamo controllare il tipo di modello costruito da questa classe 
# tramite il parametro loss specificando il tipo di funzione di costo,
# inserendo come funzione di costo 'log' andremo a creare una regressione logistica
# che utilizza lo Stochastic Gradient Descend durante l'addestramento 
# come algoritmo di ottimizzazione. 

# Prima di ogni epoca è necessario mescolare il set di addestramento
# per evitare dei cicli durante l'addestramento, per farlo è necessario impostare
# il parametro shuffle a 'true' che è comunque il valore di default. 

sgd.fit(X_train,Y_train)

loss = log_loss(Y_test,sgd.predict_proba(X_test))
print("LOSS: %.4f" % (loss))

# %% 
# La classe SGDClassifier ci mette anche a disposizione un metodo partila fit
# che possiamo utilizzare per eseguire un singolo step del Gradient Descend.

def minibatchGD(label,train_set, test_set, n_batches=1, epochs=10):
    X_train, Y_train = train_set 
    X_test, Y_test = test_set 
    batch_size = X_train.shape[0]/n_batches 

    sgd = SGDClassifier(loss='log')
    sgd_loss = []
    for epoch in range(epochs):
        X_shuffled, Y_shuffled = shuffle(X_train, Y_train)
        for batch in range(n_batches):
            batch_start = int(batch*batch_size)
            batch_end = int((batch+1)*batch_size)
            X_batch = X_shuffled[batch_start:batch_end,:]
            Y_batch = Y_shuffled[batch_start:batch_end]
            classes=np.unique(Y_train)
            sgd.partial_fit(X_batch, Y_batch, classes=classes)
            loss = log_loss(Y_test, sgd.predict_proba(X_test), labels=classes)
            sgd_loss.append(loss)
        print("%s Loss all'epoca %s = %.4f" % (label,epoch+1,loss))
    return (sgd, sgd_loss)

# Utilizziamo questa funziona che abbiamo creato per eseguire esattamente un Full Batch Gradient Descend
full_gd , full_gd_loss = minibatchGD('FULL BATCH GD',(X_train,Y_train),(X_test,Y_test), n_batches=1, epochs=200)

# Eseguiamo, sempre sfruttando la funziona che abbiamo creato, lo Stochastic Gradient Descend 
# ponendo il numero di batch pari al numero di esempi del train set,
# in modo che ogni step del Gradient Descend corrisponda ad un passaggio su un unico esempio del train set. 
sgd, sgd_loss = minibatchGD('STOCHASTIC GD',(X_train,Y_train),(X_test,Y_test), n_batches=X_train.shape[0], epochs=5)

# Adesso rifacciamo la stessa cosa per un Mini Batch Gradient Descend 
mini_gd , mini_gd_loss = minibatchGD('MINI BATCH',(X_train,Y_train),(X_test,Y_test), n_batches=10, epochs=50)


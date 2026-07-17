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

# Per eseguire lo Stochastic Gradient Descend abbiamo a disposizione la classe:
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss="log_loss")

# Possiamo controllare il tipo di modello costruito da questa classe 
# tramite il parametro loss specificando il tipo di funzione di costo,
# inserendo come funzione di costo 'log_loss' andremo a creare una regressione logistica
# che utilizza lo Stochastic Gradient Descend durante l'addestramento 
# come algoritmo di ottimizzazione. 

# Prima di ogni epoca è necessario mescolare il set di addestramento
# per evitare dei cicli durante l'addestramento, per farlo è necessario impostare
# il parametro shuffle a 'True' che è comunque il valore di default. 

sgd.fit(X_train,Y_train)

loss = log_loss(Y_test,sgd.predict_proba(X_test))
print("LOSS: %.4f" % (loss))

# %% 
# La classe SGDClassifier ci mette anche a disposizione un metodo partial_fit
# che possiamo utilizzare per eseguire un singolo step del Gradient Descend.

def minibatchGD(label, train_set, test_set, n_batches=1, epochs=10):
    X_train, Y_train = train_set 
    X_test, Y_test = test_set 
    batch_size = X_train.shape[0] / n_batches 

    sgd = SGDClassifier(loss='log_loss')
    sgd_loss = []
    
    # OTTIMIZZAZIONE: Calcoliamo le classi uniche una sola volta prima dei cicli
    classes = np.unique(Y_train)
    
    for epoch in range(epochs):
        X_shuffled, Y_shuffled = shuffle(X_train, Y_train)
        
        for batch in range(n_batches):
            batch_start = int(batch * batch_size)
            batch_end = int((batch + 1) * batch_size)
            
            X_batch = X_shuffled[batch_start:batch_end, :]
            Y_batch = Y_shuffled[batch_start:batch_end]
            
            sgd.partial_fit(X_batch, Y_batch, classes=classes)
            loss = log_loss(Y_test, sgd.predict_proba(X_test), labels=classes)
            sgd_loss.append(loss)
            
        print("%s Loss all'epoca %s = %.4f" % (label, epoch + 1, loss))
        
    return (sgd, sgd_loss)

# Utilizziamo questa funzione che abbiamo creato per eseguire esattamente un Full Batch Gradient Descend
full_gd , full_gd_loss = minibatchGD('FULL BATCH GD', (X_train, Y_train), (X_test, Y_test), n_batches=1, epochs=200)

# Eseguiamo, sempre sfruttando la funzione che abbiamo creato, lo Stochastic Gradient Descend 
# ponendo il numero di batch pari al numero di esempi del train set,
# in modo che ogni step del Gradient Descend corrisponda ad un passaggio su un unico esempio del train set. 
sgd, sgd_loss = minibatchGD('STOCHASTIC GD', (X_train, Y_train), (X_test, Y_test), n_batches=X_train.shape[0], epochs=5)

# Adesso rifacciamo la stessa cosa per un Mini Batch Gradient Descend 
mini_gd , mini_gd_loss = minibatchGD('MINI BATCH GD', (X_train, Y_train), (X_test, Y_test), n_batches=10, epochs=50)

# %%
# Confrontiamo in un grafico le fluttuazioni delle funzioni di costo
# di questi tre ottimizzatori.

plt.rcParams['figure.figsize'] = (14,8)

plt.plot(sgd_loss, label="STOCHASTIC")
plt.plot(mini_gd_loss, label="MINI BATCH")
plt.plot(full_gd_loss, label="FULL BATCH")

# Limitiamo la visualizzazione a 200 iterazioni:
plt.xlim(xmin=0,xmax=200)

# Questo comando fa apparire il riquadro con i nomi delle linee!
plt.legend(loc="upper right", fontsize=12)

# Rendiamo il grafico leggibile aggiungendo titolo e assi
plt.title("Confronto Funzioni di Costo (Loss)", fontsize=16)
plt.xlabel("Step di ottimizzazione (Aggiornamenti dei pesi)", fontsize=12)
plt.ylabel("Log Loss", fontsize=12)

# %%

# Le sole 200 iterazioni non consentono allo stochastic di convergere,
# mentre mini batch e full batch convergono allo stesso valore della funzione di costo.
# Se preferire l'uno o l'altro: solitamente il mini batch è più consigliabile
# ma dipende dalla dimensione del dataset.
# Sicuramente per dataset molto grandi è impossibile utilizzare il full batch
# perchè l'intero dataset viene caricato in memoria e quindi un approccio che utilizza 
# il mini batch è sicuramente migliore.
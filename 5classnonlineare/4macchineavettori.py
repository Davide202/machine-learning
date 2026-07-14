# Modello di classificazione lineare: Macchine a vettore di supporto
# SVM = Support Vector Machines 

# Le macchine SVM affrontano il problema della classificazione 
# in maniera differente rispetto a quanto visto fin ora.
# Infatti lo scopo di un SVM è trovare una retta 
# che massimizza il margine tra le classi,
# cioè lo spazio che le separa.
# Lo fa utilizzando una minuscola parte del dataset che vengono chiamati vettori di supporto,
# tutti gli altri esempi sono ininfluenti per il modello.

# Questo approccio offre alcuni vantaggi interessanti:
# maggiore resistenza agli outliers che non influiscono affatto.

# Quindi tutto che importa ad una SVM è riuscire a massimizzare questo margine,
# perchè un margine maggiore, che separa le classi, corriponde ad un errore
# di generalizzazione minore e quindi ad un modello migliore. 

# Quello che fa una regressione logistica è utilizzare gli esempi del train set 
# che meglio rappresentano le classi, quindi l'esempio ideale per ogni classe. 
# Al contrario, una macchina SVM impara dagli esempi che creano più confusione.
# Utilizza solo questi esempi per creare il modello.

# Quindi una SVM utilizza un approccio all'apprendimento,
# che può essere riassunto nella seguente frase:
# Se riesco a classificare i casi più difficili allora 
# classificare gli altri sarà una passeggiata.


# %% 
# Costruiamo una macchina SVM per classificare i fiori dell'iris dataset
# in base alla classe di appartenenza.


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import LinearSVC 

iris_path = "../data/iris.csv"

iris = pd.read_csv(
    iris_path,
    header=0,
    names=["sepal.length","sepal.width","petal.length","petal.width","class"]
)

iris.info()
iris.head()

# %%

X = iris.drop("class",axis=1).values 
Y = iris["class"].values 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)

# Encoding
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Standardizzazione 
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# %%
# Creiamo un sotto-dataset contenente soltanto due proprietà

X2_train = X_train[:,:2]
X2_test = X_test[:,:2]

svc = LinearSVC(C=1.0)
svc.fit(X2_train,Y_train)

accuracy_train = svc.score(X2_train,Y_train)
accuracy_test = svc.score(X2_test,Y_test)

print("ACCURACY con 2 proprietà: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
# Questo modello presenta overfitting, 
# potremmo provare a migliorarlo modificando il parametro di regolarizzazione C

# %%
# Importiamo la funzione dal nostro nuovo file viz_svm.py
from viz_svm import plot_decision_boundaries

# Disegniamo i confini di decisione usando le 2 proprietà (X2_train)
plot_decision_boundaries(svc, X2_train, Y_train, title="SVM Lineare - Confini di Decisione (2 Proprietà)")



# %%
# Costruiamo il modello con tutte le proprietà 


# Aumentiamo max_iter per garantire che l'algoritmo converga senza mostrare avvisi rossi (Warning)
svc_all = LinearSVC(C=1.0, max_iter=10000)
svc_all.fit(X_train, Y_train)

accuracy_train_all = svc_all.score(X_train, Y_train)
accuracy_test_all = svc_all.score(X_test, Y_test)

print("ACCURACY con TUTTE le proprietà: TRAIN=%.4f TEST=%.4f" % (accuracy_train_all, accuracy_test_all))

# Come potrai notare eseguendo la cella, fornendo tutte e 4 le proprietà 
# l'accuratezza sul Test Set salirà drasticamente rispetto all'uso di sole 2 feature!



# %%

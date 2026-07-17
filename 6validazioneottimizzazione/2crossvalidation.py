# Eseguiamo una K-Fold cross validation
# per validare un modello di machine learning
# prima di andarlo a testare sul test set.



# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris_path = "../data/iris.csv"

iris = pd.read_csv(
    iris_path,
    header=0,
    names=["sepal length","sepal width","petal length","petal width","class"]
)
iris.head()


# Creiamo i set di addestramento e test

X = iris.drop("class",axis=1).values  
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)

lr = LogisticRegression(max_iter=1000)


# %%
# PRIMO METODO: KFold standard
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10,random_state=1,shuffle=True)

scores = [] # array per i valori dell'accuracy

for k , (train, test) in enumerate(kfold.split(X_train)):
    
    lr.fit(X_train[train],Y_train[train])
    
    score = lr.score(X_train[test],Y_train[test])
    
    scores.append(score)

    print("FOLD %d: Accuracy=%.4f" % (k,score))

accuracy = np.array(scores).mean()
print("Validation accuracy = %.4f" % accuracy)

# %%
# SECONDO METODO: StratifiedKFold
# C'è un problema: creando i Folds tramite la classe KFold 
# il numero di esempi appartenenti ad una determinata classe 
# all'interno di ogni Fold potrebbe essere sbilanciato.
# Questo sbilanciamento tra le classi potrebbe creare problemi durante l'addestramento.
# Per ovviare a questo problema sklearn ci mette a disposizione la classe StratidiedKFold. 

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

scores = [] # array per i valori dell'accuracy

for k , (train, test) in enumerate(kfold.split(X_train,Y_train)):
    
    lr.fit(X_train[train],Y_train[train])
    
    score = lr.score(X_train[test],Y_train[test])
    
    scores.append(score)

    print("FOLD %d: Accuracy=%.4f" % (k,score))

accuracy = np.array(scores).mean()
print("Validation accuracy = %.4f" % accuracy)


# %%
# TERZO METODO: cross_val_score (La via più rapida ed efficiente)
# Un'altra soluzione è utilizzare la KFold cross validation come fosse una funzione di score.
# Nota: per problemi di classificazione, cross_val_score usa automaticamente StratifiedKFold sotto il cofano!

from sklearn.model_selection import cross_val_score 

lr = LogisticRegression(max_iter=1000)

score = cross_val_score(lr,X_train,Y_train, cv=10)

print("MEAN SCORE = %.4f" % (score.mean()))

lr.fit(X_train,Y_train)

test_accuracy = lr.score(X_test, Y_test)
print("TEST ACCURACY FINALE SUI NUOVI DATI = %.4f" % test_accuracy)


# %%

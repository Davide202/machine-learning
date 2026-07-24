

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import time 

iris_path = "../data/iris.csv"

iris = pd.read_csv(
    iris_path,
    header=0,
    names=["sepal length","sepal width","petal length","petal width","class"]
)
iris.head()

# %%
# Creiamo i set di addestramento e test

X = iris.drop("class",axis=1).values  
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)

print("X_train: n-esempi = " + str(X_train.shape[0]) + " su n-classi = " + str(X_train.shape[1]))
print("X_test: n-esempi = " + str(X_test.shape[0]) + " su n-classi = " + str(X_test.shape[1]))

# %%

# Per utilizzare la GridSearch con sklearn possiamo utilizzare la classe GridSearchCV 
# che oltre eseguire la gridsearch si occuperà internamente di validare il modello 
# tramite cross-validation
# mentre come modello utilizzeremo una macchina a vettori di supporto

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 

svc = SVC()

# Creiamo un dizionario con i valori degli iper parametri che vogliamo andare a testare
params = {
    "kernel": ["linear","rbf","sigmoid","poly"],
    "gamma": [0.1, 1, "auto"],
    "C": [1, 10, 100, 1000]
}

gs = GridSearchCV(svc, params, cv=10) # impostiamo come numero di default 10 

start_time = time.time()
gs.fit(X_train,Y_train)
print("Tempo trascorso per l'addestramento della GridSearch :: ", (time.time() - start_time) )

# La grid search ha trovato la combinazione ottimale degli iper parametri in meno di un secondo!

# Vediamo quali sono i valori degli iper parametri che la grid search ha identificato 

# gs.best_params_ # {'C': 1, 'gamma': 0.1, 'kernel': 'linear'}
# gs.best_score_ # np.float64(0.9818181818181818)

# print("BEST PARAMS %s - BEST SCORE %.4f" % (gs.best_params_, gs.best_score_))
print(f"GridSearch BEST PARAMS {gs.best_params_} - BEST SCORE {gs.best_score_:.4f}")

# Piuttosto che addestrare nuovamente il modello con questi iper parametri
# possiamo recuperarlo direttamente dalla grid search 

svc = gs.best_estimator_

print("GridSearch SVC SCORE: ", svc.score(X_test, Y_test) )


# Andiamo a vecedere la Random Search

from sklearn.model_selection import RandomizedSearchCV 

svc = SVC()

params = {
    "kernel": ["linear","rbf","sigmoid","poly"],
    "gamma": [0.1, 1, "auto"],
    "C": [1, 10, 100, 1000]
}

rs = RandomizedSearchCV(svc,params, cv=10)

start_time = time.time()
rs.fit(X_train,Y_train)
print("Tempo trascorso per l'addestramento della RandomizedSearch :: ", (time.time() - start_time) )

print(f"RandomizedSearch BEST PARAMS {rs.best_params_} - BEST SCORE {rs.best_score_:.4f}")

svc = rs.best_estimator_
print("RandomizedSearch SVC SCORE: ", svc.score(X_test, Y_test) )


# %%

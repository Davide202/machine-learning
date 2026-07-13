# Foreste Casuali :: Random Forest


# Apprendimento Ensamble, si basa sulla combinazione di più modelli per ottenere un modello più robusto, 
# otteniamo migliore generalizzazione e minor overfitting.

# Una random forest è un modello che usa l'apprendimento ensable, 
# mettendo insieme più alberi decisionali.

# Costruire una Random Forest:
# 1 - Scegliamo un numero di alberi per la nostra foresta
# 2 - Prendiamo un numero di esempi a caso dal dataset
# 3 - Addestriamo un albero di questo sottoinsieme
# 4 - Ripetiamo i punti 2 e 3 fino ad avere costruito tutti gli alberi
# 5 - Per eseguire una predizione, utilizziamo tutti gli alberi 
#       e poi facciamo la media delle varie predizioni.

# Il grosso vantaggio della Random Forest è che ha il numero minimo di iperparametri.
# Infatti l'unico parametri di cui dobbiamo veramente preoccuparci è il numero di alberi.




# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 


titanic_path = '../data/titanic.csv'
titanic = pd.read_csv(titanic_path)
titanic = titanic.drop('Name',axis=1)
titanic = pd.get_dummies(titanic)
SURVIVED = 'Survived'
X = titanic.drop(SURVIVED,axis=1).values 
Y = titanic[SURVIVED].values 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# %%
# Andiamo a costruire la nostra foresta
forest = RandomForestClassifier(random_state=False)
forest.fit(X_train,Y_train)

Y_pred_train = forest.predict(X_train)
Y_pred = forest.predict(X_test)

accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test)) # ACCURACY: TRAIN=0.8935 TEST=0.8202
# Il nostro modello soffre di Overfitting,
# Andiamo a limitare la profondità massima degli alberi

# %%
forest = RandomForestClassifier(random_state=False,max_depth=8,n_estimators=30)
forest.fit(X_train,Y_train)
Y_pred_train = forest.predict(X_train)
Y_pred = forest.predict(X_test)

accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))# ACCURACY: TRAIN=0.9161 TEST=0.8577


# %%

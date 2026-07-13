# In informatica un albero è una struttura dati che va letto dall'alto verso il basso,
# è composto da archi, nodi e foglie.

# Un albero decisionale è un modello di ML che utilizza una struttura dati ad albero
# per contenere le informazioni e prendere le decisioni.

# Come fa l'algoritmo a scegliere le domande per creare l'albero decisionale:
# lo fa scegliendo la domanda che massimizza il guadagno di informazioni.

# L'obiettivo dell'addestramento è giungere alle foglie 
# ponendo il minor numero di domande possibile.

# Metriche per il calcolo delle impurità: GINI e ENTROPIA.

# Una maggiore profondità dell'albero aumeta la complessità del modello 
# e il pericolo di overfitting.

# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

titanic_path = '../data/titanic.csv'
titanic = pd.read_csv(titanic_path)

titanic.info()

titanic.head()



# %%

# Rimuoviamo la proprietà Name perchè non porta nessuna informazione significativa per il modello.
titanic = titanic.drop('Name',axis=1)

# Eseguiamo il "One-Hot Encoding" 
# (che significa accendere "a 1" solo la colonna corretta e lasciare "a 0" le altre) 
# sulla proprietà Sex per crearci delle variabili di comodo.
titanic = pd.get_dummies(titanic)

titanic.head()


# %%
# Creiamo gli array Numpy per proprietà e target

SURVIVED = 'Survived'

X = titanic.drop(SURVIVED,axis=1).values 
Y = titanic[SURVIVED].values 

# Creiamo gli array per addestramento e test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

X_train.shape # (620, 7) 620 esempi e 7 proprità per addestrare il modello

# %%
# Un grosso vantaggio degli alberi decisionali è che non richiedono che i dati
# siano sulla stessa scala, quindi non serve fare normalizzazione o standardizzazione.

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini')
tree.fit(X_train,Y_train)

Y_pred_train = tree.predict(X_train)
Y_pred = tree.predict(X_test)

accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test)) # ACCURACY: TRAIN=0.9806 TEST=0.7715
# Quindi il nostro albero soffre di overfitting questo vuol dire che è troppo complesso,
# la complessità di un albero è legata alla sua profondità.
# Proviamo a ridurre la profondità dell'albero e vediamo se riusciamo a ridurre l'overfitting.


# %%
# Possiamo ridurre la profondità dell'albero tramite il parametro max_depth

tree = DecisionTreeClassifier(criterion='gini',max_depth=6)
tree.fit(X_train,Y_train)

Y_pred_train = tree.predict(X_train)
Y_pred = tree.predict(X_test)

accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test,Y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test)) # ACCURACY: TRAIN=0.8935 TEST=0.8202


# %%
# Un'altra caratteristica fantastica degli alberi è che possono essere facilmente visualizzati.

from sklearn.tree import export_graphviz
from viz_sklearn import salva_albero_da_modello

file_path = "tree.dot"
colonne_features = titanic.columns.drop(SURVIVED)

dotfile = open(file_path,"w")
export_graphviz(tree,out_file=dotfile, feature_names=colonne_features)
dotfile.close()

# Stampa dell'albero in file:
salva_albero_da_modello(tree, colonne_features, "albero_titanic_matplotlib.png")

# Per visualizzare l'albero possiamo anche incollare il contenuto del file nel sito web: webgraphviz.com








# %%
from viz_graphviz import salva_albero_da_dot

# Gli passi il file che hai appena salvato con export_graphviz
salva_albero_da_dot(file_path, "il_mio_albero_titanic")
# %%

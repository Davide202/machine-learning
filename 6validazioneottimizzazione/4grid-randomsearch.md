
---

# Ottimizzazione degli Iperparametri: Grid Search vs Randomized Search

Questo documento analizza l'implementazione e il confronto di due tecniche fondamentali per l'ottimizzazione degli iperparametri di un modello di Machine Learning: la **Grid Search** e la **Randomized Search**.

Il modello scelto per questo test è una **Support Vector Machine (SVC)** addestrata sul celebre dataset Iris.

---

## 1. Preparazione dei Dati

Il primo step consiste nel caricare il dataset, separare le feature (`X`) dalle etichette (`Y`) e suddividere i dati in un set di addestramento e uno di test.

```python
import pandas as pd
import numpy as np
import time 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

# Caricamento del dataset
iris_path = "../data/iris.csv"
iris = pd.read_csv(iris_path, header=0, names=["sepal length","sepal width","petal length","petal width","class"])

# Separazione feature e target
X = iris.drop("class", axis=1).values  
Y = iris["class"].values

# Split in Train e Test (75% - 25% di default)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

```

---

## 2. Definizione della Griglia degli Iperparametri

Per la nostra Support Vector Machine (SVC) vogliamo testare diverse architetture variando tre parametri chiave:

* **Kernel:** La funzione matematica usata per separare i dati (`linear`, `rbf`, `sigmoid`, `poly`).
* **Gamma:** Regola quanto i singoli esempi influenzano il confine di decisione (`0.1`, `1`, `"auto"`).
* **C:** Il parametro di regolarizzazione, che bilancia il margine di errore e la precisione del confine di decisione (`1`, `10`, `100`, `1000`).

```python
params = {
    "kernel": ["linear", "rbf", "sigmoid", "poly"],
    "gamma": [0.1, 1, "auto"],
    "C": [1, 10, 100, 1000]
}

```

> 🧮 **Nota:** Questa griglia genera esattamente **48 combinazioni totali** (4 kernel × 3 gamma × 4 valori di C).

---

## 3. L'approccio GridSearchCV

La classe `GridSearchCV` è esaustiva: prova **tutte e 48 le combinazioni possibili**, eseguendo per ciascuna una Cross Validation a 10 Fold.

```python
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(SVC(), params, cv=10) 

start_time = time.time()
gs.fit(X_train, Y_train)
tempo_gs = (time.time() - start_time) * 10

print(f"Tempo GridSearch: {tempo_gs}")
print(f"GridSearch BEST PARAMS {gs.best_params_} - BEST SCORE {gs.best_score_:.4f}")

modello_gs = gs.best_estimator_
print(f"GridSearch TEST SCORE: {modello_gs.score(X_test, Y_test):.4f}")

```

---

## 4. L'approccio RandomizedSearchCV

La classe `RandomizedSearchCV` è invece stocastica: piuttosto che provare tutte e 48 le combinazioni, ne seleziona a caso un sottoinsieme. Di default, il parametro `n_iter` è impostato a **10**, il che significa che esplorerà solo 10 combinazioni casuali.

```python
from sklearn.model_selection import RandomizedSearchCV 

rs = RandomizedSearchCV(SVC(), params, cv=10) # n_iter=10 di default

start_time = time.time()
rs.fit(X_train, Y_train)
tempo_rs = (time.time() - start_time) * 10

print(f"Tempo RandomSearch: {tempo_rs}")
print(f"RandomSearch BEST PARAMS {rs.best_params_} - BEST SCORE {rs.best_score_:.4f}")

modello_rs = rs.best_estimator_
print(f"RandomSearch TEST SCORE: {modello_rs.score(X_test, Y_test):.4f}")

```

---

## 5. Osservazioni e Analisi dei Risultati

Analizzando l'output dell'esecuzione, emergono alcuni dettagli tecnici fondamentali.

### A. Tempi di esecuzione

I due approcci mostrano un'importante differenza di performance computazionale:

* **Grid Search:** Esplorando tutte e 48 le combinazioni (eseguite in 10 fold), ha richiesto logicamente un tempo di addestramento maggiore.
* **Random Search:** Avendo esplorato solo 10 combinazioni casuali, si rivela nettamente più veloce, pur riuscendo a trovare un modello finale con un'accuratezza equivalente. Su dataset massivi, questo approccio permette di abbattere drasticamente i tempi di ricerca.

### B. Il mistero dei parametri differenti e il Kernel Lineare

Eseguendo lo script, è comune ottenere due combinazioni vincenti apparentemente diverse:

* *Esempio Grid Search:* `{'C': 1, 'gamma': 0.1, 'kernel': 'linear'}`
* *Esempio Random Search:* `{'kernel': 'linear', 'gamma': 'auto', 'C': 10}`

Eppure, entrambe restituiscono lo stesso identico punteggio sul Test Set. Il motivo risiede nella matematica delle SVM: quando il parametro `kernel` viene impostato su `'linear'`, l'algoritmo traccia una retta (o iperpiano) e **il parametro `gamma` viene matematicamente ignorato** (poiché serve solo per la mappatura non lineare).
Di conseguenza, che il valore di `gamma` sia `0.1`, `1` o `'auto'`, il modello si comporterà nello stesso identico modo. La Grid Search restituisce la prima combinazione lineare ottimale testata, mentre la Random Search restituisce quella pescata casualmente.

### C. La sintassi 'auto'

Quando i risultati riportano `'auto'` come valore di `gamma`, non significa che manca un dato. La stringa `"auto"` è stata passata esplicitamente nel dizionario `params`: è una *keyword* di Scikit-Learn che fa dedurre automaticamente al modello il miglior valore di gamma calcolandolo in base al numero di feature del dataset.

# 
# Ottimizzeremo un **Percettrone Multistrato (MLP)**. 
# Utilizzeremo la classe `RandomizedSearchCV`, 
# che è fantastica perché unisce in un colpo solo sia la ricerca casuale degli iperparametri, 
# sia la **K-Fold Cross Validation** per valutare ogni combinazione!
# 
# Per generare numeri casuali in range specifici (es. tra 10^-4 e 1), utilizzeremo la libreria `scipy.stats`.
# 
# ### Lo Script: Ottimizzazione con Random Search


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from scipy.stats import loguniform

# 1. Prepariamo un dataset fittizio per l'esempio
X, Y = make_classification(n_samples=1500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. Definiamo il modello "base"
# Impostiamo un max_iter alto per evitare i ConvergenceWarning
mlp = MLPClassifier(max_iter=1000, random_state=42)

# 3. Definiamo la GRIGLIA degli iperparametri per la Random Search
# Invece di un singolo valore, passiamo liste o distribuzioni statistiche
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], # Diverse architetture dei nodi
    'activation': ['relu', 'logistic', 'tanh'],                 # Diverse funzioni di attivazione
    'alpha': loguniform(1e-4, 1e-1),                            # Regolarizzazione (L2) tra 10^-4 e 10^-1
    'learning_rate_init': loguniform(1e-4, 1e-1)                # Learning rate iniziale
}

# 4. Inizializziamo il RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_dist,
    n_iter=20,          # Il numero MAGICO: proverà solo 20 combinazioni scelte a caso!
    cv=5,               # Applica una K-Fold Cross Validation con K=5 per ogni tentativo
    n_jobs=-1,          # Usa tutti i core del PC per fare prima
    verbose=2,          # Stampa i log per farci vedere a che punto è
    random_state=42
)

# 5. Avviamo l'addestramento e la ricerca
print("Inizio la Random Search. Preparati, ci vorrà un po'...")
random_search.fit(X_train, Y_train)

# 6. Stampiamo i risultati trionfali!
print("\n--- RICERCA COMPLETATA ---")
print("I migliori iperparametri trovati sono:")
print(random_search.best_params_)

print("\nAccuratezza media in Cross Validation con questi parametri: %.4f" % random_search.best_score_)

# 7. Test finale sul Test Set "chiuso in cassaforte"
# best_estimator_ contiene già il modello riaddestrato sull'intero Train Set con i parametri perfetti (lo step 5 dei tuoi appunti!)
modello_perfetto = random_search.best_estimator_
test_accuracy = modello_perfetto.score(X_test, Y_test)

print("Accuratezza finale sul Test Set: %.4f" % test_accuracy)


# ### Cosa sta succedendo dietro le quinte?
# 
# Questa classe fa tutto il "lavoro sporco" descritto nei tuoi appunti in sole 3 fasi:
# 
# 1. **Estrazione casuale (`n_iter=20`):** Invece di provare centinaia di combinazioni possibili 
#       (che richiederebbero ore o giorni, come accade nella *Grid Search*), 
#       l'algoritmo pesca a caso 20 combinazioni dalla scatola `param_dist`.
# 2. **K-Fold CV Automatica (`cv=5`):** Per ognuna di quelle 20 combinazioni, 
#       esegue una Cross Validation a 5 Fold sul Train Set. Significa che, in totale, 
#       addestrerà la rete neurale **100 volte** (20 combinazioni moltiplicate per 5 fold).
# 3. **Il riaddestramento automatico:** Quando finisce: 
#       prende i parametri vincenti, butta via i calcoli parziali, 
#       e **riaddestra in automatico un nuovo modello finale** usando il 100% di `X_train`. 
#       Questo modello pronto all'uso è salvato e accessibile chiamando `random_search.best_estimator_`.
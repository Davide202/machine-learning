# Vediamo delle tecniche per ottimizzare gli iperparametri di un modello di machine learning.

# Gli iperparametri sono tutte le variabili che vanno specificate
# prima dell'addestramento e che definiscoo l'architettura di un modello.

# Per il Gradient Descend e quindi per tutti i modelli che se ne servono per l'addestramento dobbiamo specificare:
# - il numero di epoche cioè in numero di volte che il Gradient Descend passa su tutto il train set
# - learning rate (anche conosciuto come alpha) nel range tra 10^-4 e 1 (ma non solo)
# - tolleranza (early stopping) cioè se vogliamo fermare in anticipo il GD nel caso in cui ad ogni iterazione
#   il valore della funzione di costo non diminuisce di un piccolo valore prefissato, allora dobbiamo impostare la tolleranza.

# Per il Mini Batch Gradient Descend dobbiamo specificare anche questi parametri:
# - numero di iterazioni: numero di volte che il GD viene eseguito (numero di epoche x numero di batch)
# - numero di batches, batches in cui dividere il dataset
# - batch size, numero di esempi all'interno di un singolo batch.

# Se vogliamo applicare la Regolarizzazione dobbiamo specificare:
# - tipo di regolarizzazione: L1, L2, entrambe
# - parametro di regolarizzazione (lambda, C = 1/lambda) tra 10^-4 e 10 (ma non solo).

# Per gli alberi decisionali dobbiamo specificare:
# - profondità massima, tenendo in mente che una maggiore profondità corrisponde ad una maggiore complessità che aumenta il rischio di overfitting.
# - metrica di impurità: Gini, entropia

# Per il K-Nearest Neighbors dobbiamo specificare:
# - numero di vicini (K), un buon valore è 5, un valore di K troppo basso aumenta il rischio di overfitting. 

# Per una 

---

# Ottimizzazione degli Iperparametri

Vediamo delle tecniche per ottimizzare gli iperparametri di un modello di machine learning.

Gli iperparametri sono tutte le variabili che vanno specificate prima dell'addestramento e che definiscono l'architettura di un modello.

---

### 1. Iperparametri per il Gradient Descent

Per il Gradient Descent (e quindi per tutti i modelli che se ne servono per l'addestramento) dobbiamo specificare:

* **Il numero di epoche:** cioè il numero di volte che il Gradient Descent passa su tutto il train set.
* **Learning rate:** (anche conosciuto come $\alpha$) nel range tra $10^{-4}$ e $1$ (ma non solo).
* **Tolleranza (Early Stopping):** cioè se vogliamo fermare in anticipo il GD nel caso in cui ad ogni iterazione il valore della funzione di costo non diminuisce di un piccolo valore prefissato, allora dobbiamo impostare la tolleranza.

Per il **Mini Batch Gradient Descent** dobbiamo specificare anche questi parametri:

* **Numero di iterazioni:** numero di volte che il GD viene eseguito (numero di epoche $\times$ numero di batch).
* **Numero di batches:** batches in cui dividere il dataset.
* **Batch size:** numero di esempi all'interno di un singolo batch.

> 💡 **Considerazione sul Learning Rate:** Il Learning Rate è l'iperparametro più critico in assoluto. Un valore troppo alto farà divergere l'algoritmo (la loss esplode), mentre un valore troppo basso lo renderà lentissimo. Spesso si usano tecniche di *Learning Rate Scheduling*, che partono con un valore alto per poi abbassarlo man mano che ci si avvicina al minimo.

---

### 2. Iperparametri per modelli specifici

**Se vogliamo applicare la Regolarizzazione dobbiamo specificare:**

* **Tipo di regolarizzazione:** L1, L2, entrambe (Elastic Net).
* **Parametro di regolarizzazione:** ($\lambda$, oppure $C = \frac{1}{\lambda}$) solitamente impostato tra $10^{-4}$ e $10$ (ma non solo).

> 🔍 **Focus L1 vs L2:** Ricorda che la regolarizzazione L1 (Lasso) tende ad azzerare i pesi delle feature inutili (fa *feature selection*), mentre la L2 (Ridge) riduce proporzionalmente tutti i pesi senza azzerarli, evitando che un singolo peso diventi troppo dominante.

**Per gli Alberi Decisionali dobbiamo specificare:**

* **Profondità massima (max depth):** tenendo in mente che una maggiore profondità corrisponde ad una maggiore complessità che aumenta il rischio di overfitting.
* **Metrica di impurità:** Gini, entropia.

**Per il K-Nearest Neighbors dobbiamo specificare:**

* **Numero di vicini (K):** un buon valore è 5. Un valore di K troppo basso aumenta il rischio di overfitting (il modello diventa troppo sensibile al rumore locale).

**Per una Kernel SVM dobbiamo specificare:**

* **La funzione Kernel:** Gaussiana (RBF), Sigmoidale, Polinomiale, custom.
* **Sigma (o gamma):** che controlla la sensibilità alla differenza tra gli esempi.

**Nel caso del Percettrone Multistrato bisogna specificare:**

* **Numero di hidden layers:** numero di layers intermedi tra quello di input e quello di output.
* **Numero di nodi per hidden layers:** gli hidden layers possono contenere un numero di nodi differente tra loro.
* **Funzione di attivazione per (ogni) hidden layers:** una buona tecnica per problemi di classificazione è usare **ReLU** per gli hidden layer e la **sigmoide** per l'output layer.

> 📝 **Nota sull'Output Layer:** La Sigmoide è perfetta se il problema è una classificazione binaria (2 classi). Se invece stai facendo una classificazione multiclasse (come il dataset MNIST con 10 cifre), la funzione standard per l'output layer è la **Softmax**.

---

### 3. Ricerca degli Iperparametri

Fin ora quello che abbiamo fatto è scegliere i valori degli iperparametri, addestrare il modello e se questo non va bene scegliere manualmente altri valori e riaddestrarlo (**Manual Search**).

Una tecnica migliore è la **Grid Search**: definiamo diversi valori per diversi iperparametri e proviamo *tutte* le combinazioni. Questa tecnica non è molto efficiente e può richiedere molto tempo.

Un'altra tecnica è la **Random Search**: siccome di solito gli iperparametri non hanno la stessa importanza nel processo di addestramento, questo approccio si rivela spesso più rapido e più efficiente nel trovare la giusta combinazione di iperparametri rispetto alla Grid Search.

> 🛠️ **Implementazione in Python:** Scikit-Learn ti mette a disposizione due classi comodissime per eseguire queste operazioni in modo automatico, combinandole direttamente con la Cross Validation: si chiamano `GridSearchCV` e `RandomizedSearchCV`.

---





----------------------------------------------------------------------------------------------------------------------
APPUNTI ORIGINALI
Vediamo delle tecniche per ottimizzare gli iperparametri di un modello di machine learning.

Gli iperparametri sono tutte le variabili che vanno specificate
prima dell'addestramento e che definiscoo l'architettura di un modello.

Per il Gradient Descend e quindi per tutti i modelli che se ne servono per l'addestramento dobbiamo specificare:
- il numero di epoche cioè in numero di volte che il Gradient Descend passa su tutto il train set
- learning rate (anche conosciuto come alpha) nel range tra 10^-4 e 1 (ma non solo)
- tolleranza (early stopping) cioè se vogliamo fermare in anticipo il GD nel caso in cui ad ogni iterazione
  il valore della funzione di costo non diminuisce di un piccolo valore prefissato, allora dobbiamo impostare la tolleranza.

Per il Mini Batch Gradient Descend dobbiamo specificare anche questi parametri:
- numero di iterazioni: numero di volte che il GD viene eseguito (numero di epoche x numero di batch)
- numero di batches, batches in cui dividere il dataset
- batch size, numero di esempi all'interno di un singolo batch.

Se vogliamo applicare la Regolarizzazione dobbiamo specificare:
- tipo di regolarizzazione: L1, L2, entrambe
- parametro di regolarizzazione (lambda, C = 1/lambda) tra 10^-4 e 10 (ma non solo).

Per gli alberi decisionali dobbiamo specificare:
- profondità massima, tenendo in mente che una maggiore profondità corrisponde ad una maggiore complessità che aumenta il rischio di overfitting.
- metrica di impurità: Gini, entropia

Per il K-Nearest Neighbors dobbiamo specificare:
- numero di vicini (K), un buon valore è 5, un valore di K troppo basso aumenta il rischio di overfitting. 

Per una Kernel SVM dobbiamo specificare:
- la funzione Kernel, Gaussiana (RBF) , Sigmoidale, Polinomiale, custom
- Sigma (o gamma) che controlla la sensibilità alla differenza tra gli esempi.

Nel caso del Percettrone Multistrato bisogna specificare:
- Numero di hidden layers, numero di layers intermedi tra quello di input e quello di output
- numero di nodi per hidden layers, gli hidden layers possono contenere un numero di nodi differente tra loro
- funzione di attivazione per (ogni) hidden layers, una buona tecnica per problemi di classificazione 
    è usare ReLU per gli hidden layer e la sigmoide per l'output layer

Ricerca degli iperparametri,
fin ora quello che abbiamo fatto è scegliere i valori degli iperparametri, addestrare il modello 
e se questo non va bene scegliere manualmente altri valori e riaddestrarlo (Manual Search).

Una tecnica migliore è Grid Search: definiamo diversi valori per diversi iperparametri e proviamo tutte le combinazioni.
Questa tecnica non è molto efficiente e può richiedere molto tempo.

Un'altra tecnica è la Random Search, siccome di solito gli iperparametri non hanno la stessa importanza nel processo di addestramento,
questo approccio si rivela spesso più rapido e più efficiente nel trovare la giusta combinazione di iperparametri rispoetto alla Grid Search.






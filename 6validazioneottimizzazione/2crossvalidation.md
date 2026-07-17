
---

# Tecniche di Cross Validation

Vediamo delle tecniche di cross validation per verificare la qualità di un modello di machine learning.

### 1. Equilibrio tra Bias e Varianza

Abbiamo osservato come la suddivisione del dataset in un set di addestramento ed uno di test ci permette di ridurre l'overfitting ed ottenere un modello in grado di generalizzare bene.

* Se il modello è troppo semplice e non riesce ad apprendere le varie relazioni tra le proprietà, allora il bias è troppo alto e il modello soffre di **underfitting**.
* Al contrario quando il modello è troppo complesso e memorizza il set di addestramento piuttosto che utilizzarlo per apprendere allora la varianza è troppo alta e il modello soffre di **overfitting**.

Quando invece bias e varianza sono in equilibrio allora il modello avrà la giusta complessità e sarà in grado di generalizzare bene, cioè di eseguire predizioni abbastanza accurate su un nuovo set di dati.

> 💡 **Considerazione sul Trade-off Bias/Varianza:** Trovare questo equilibrio è il cuore del Machine Learning. Un modello in underfitting è come uno studente che non ha studiato abbastanza (sbaglierà sia gli esercizi a casa che l'esame finale); un modello in overfitting è come uno studente che ha imparato le risposte a memoria (perfetto a casa, ma fallisce su domande formulate in modo diverso).

---

### 2. I limiti dell'approccio standard

Fin ora abbiamo utilizzato i seguenti step:

* Selezionare il modello
* Addestrare sul train set
* Testare sul test set
* Se i risultati ci soddisfano abbiamo il nostro modello.

Ma nei modelli ci sono un gran numero di iperparametri da ottimizzare, ma è sbagliato sfruttare il test set per la fase di model selection e hyperparameter tuning: iterando sul test set alla ricerca del modello migliore si rischia di causare l'overfitting del modello sul test set.

> 📌 **Approfondimento:** Questo fenomeno è noto come **Ottimizzazione sul Test Set**. Più modifichiamo i parametri guardando i risultati finali del test, più stiamo indirettamente trasferendo informazioni dal Test Set al modello. Alla fine, il Test Set non sarà più un giudice imparziale, ma farà parte dell'addestramento.

---

### 3. L'approccio con Validation Set

Una soluzione è quella di utilizzare un set per la validazione del modello, cioè prendere il dataset e dividerlo in 3 sub set:

* **Train set:** per l'addestramento del modello.
* **Validation set:** per selezionare il modello.
* **Test set:** per testare il modello su dati sconosciuti.

Utilizzando il 'validation set' per la selezione del modello e il 'test set' per misurare la capacità di generalizzazione del modello, eliminiamo il rischio di 'data leakage' e possiamo verificare che il modello non soffra di overfitting, su nessuno dei set.

> 📊 **Esempio di Split Classico:** Una divisione molto comune e utilizzata in questo scenario è **60% / 20% / 20%** (Train / Validation / Test).

**Ma ci sono due problemi:**

1. Dovendo estrarre un ulteriore sub set dal dataset, andiamo a ridurre ulteriormente il train set, il che è un grosso problema se stiamo lavorando con un dataset già di per se abbastanza piccolo.
2. Il modello potrebbe dipendere da come abbiamo eseguito lo split dei tre sub set.

> ⚠️ **Considerazione sul Problema dello Split:** Se per puro caso (a causa di uno split sfortunato) tutti gli esempi "difficili" finiscono nel Validation Set, penseremo che il nostro modello sia pessimo e continueremo a modificarlo inutilmente. Se ci finiscono tutti quelli "facili", ci illuderemo di avere un modello perfetto.

---

### 4. La K-Fold Cross Validation

Esiste una soluzione migliore e si chiama **'K-Fold cross validation'**.
Funziona così, prendiamo il dataset e lo dividiamo in un numero determinato di parti chiamato appunto Fold, quello che dobbiamo fare è costruire 4 modelli con stessi iperparametri, utilizzando ogni volta una delle 4 fold come test set e le restanti come train set.

> ⚙️ **Esempio Pratico (K=4):**
> Dividiamo il Train Set in 4 blocchi (A, B, C, D). Costruiamo 4 modelli identici e li facciamo ruotare:
> * **Modello 1:** Addestra su A, B, C ➔ Valida su D
> * **Modello 2:** Addestra su A, B, D ➔ Valida su C
> * **Modello 3:** Addestra su A, C, D ➔ Valida su B
> * **Modello 4:** Addestra su B, C, D ➔ Valida su A
> *In questo modo, ogni singolo dato è stato usato sia per l'addestramento che per la validazione.*
> 
> 

Calcoliamo l'errore per ciascun modello, ricorda che i modelli sono dello stesso tipo e con gli stessi iperparametri, differiscono solo per il set di addestramento. Calcoliamo l'errore complessivo come la media degli errori dei modelli.

---

### 5. Gli Step Definitivi

La pratica migliore è utilizzare la cross validation al posto del set di validazione, ma tenere sempre un test set a parte per la valutazione finale. Quindi gli step da seguire sono i seguenti:

1. Creare un set di addestramento e uno di test
2. Selezionare tipo di modello ed iperparametri
3. Addestrare il modello tramite cross validation usando il set di addestramento
4. Eventualmente ripetere gli step 2 e 3 fino ad aver ottenuto buoni risultati, cioè un modello che ci dia una funzione di costo abbastanza bassa.
5. Riaddestrare il modello sull'intero set di addestramento
6. Testare il modello sul test set
7. Se il risultato è buono abbiamo il modello, altrimenti torniamo allo step 2.

> 💡 **Nota cruciale sullo Step 5:** Molti dimenticano questo passaggio vitale! La Cross Validation serve *solo* a trovare la configurazione ideale e calcolare un errore attendibile. Trovati i parametri, si butta via tutto e si istruisce un nuovo modello "fresco" dando in pasto alla macchina il 100% dei dati del Train Set.

---

### 6. Come scegliere il numero di Folds da utilizzare:

* Per la maggior parte dei casi **10** va bene.
* Ma se un dataset è molto grande utilizzare **5**.
* Se il dataset è piccolo utilizzare **20**.

Un caso particolare è quello di un dataset davvero piccolo, in tal caso possiamo utilizzare una tecnica chiamata **'leave one out cross validation'**, cioè lasciarne uno fuori, quindi il valore di k è pari al numero di esempi nel dataset. Quindi ad ogni ciclo della cross validation verrà utilizzato un unico esempio per il test set e tutti gli altri per l'addestramento.

> 🔍 **Considerazione Computazionale sulla LOOCV:** La tecnica Leave-One-Out è statisticamente precisissima, ma richiede un enorme sforzo computazionale. Poiché $K = N$, se hai un dataset di 1.500 righe, costringerai il sistema ad addestrare 1.500 modelli completi! Va utilizzata solo quando i dati sono talmente pochi da rendere preziosa ogni singola riga.






---------
APPUNTI ORIGINALI

Vediamo delle tecniche di cross validation 
per verificare la qualità di un modello di machine learning.

Abbiamo osservato come la suddivisione del dataset
in un set di addestramento ed uno di test
ci permette di ridurre l'overfitting ed ottenere un modello in grado di generalizzare bene.
Se il modello è troppo semplice e non riesce ad apprendere le varie relazioni tra le proprietà,
allora il bias è troppo alto e il modello soffre di underfitting.
Al contrario quando il modello è troppo complesso e memorizza il set di addestramento
piuttosto che utilizzarlo per apprendere allora la varianza è troppo alta e il modello soffre di overfitting.
Quando invece bias e varianza sono in equilibrio allora il modello avrà la giusta complessità
e sarà in grado di generalizzare bene cioè di eseguire predizioni abbastanza accurate su un nuovo set di dati. 

Fin ora abbiamo utilizzato i seguenti step:
- Selezionare il modello
- Addestrare sul train set
- Testare sul test set 
- se i risultati ci soddisfano abbiamo il nostro modello.

Ma nei modelli ci sono un gran numero di iper parametri da ottimizzare,
ma è sbagliato sfruttare il test set per la fase di model selection e iperparameters tuning,
iterando sul test set alla ricerca del modello migliore 
si rischia di causare l'overfitting del modello sul test set. 

Una soluzione è quella di utilizzare un set per la validazione del modello,
cioè prendere il dataset e dividerlo in 3 sub set:
- train set per l'addestramento del modello
- validation set per selezionare il modello
- test set per testare il modello su dati sconosciuti.

Utilizzando il 'validation set' per la selezione del modello e il 'test set' per misurare
la capacità di generalizzazione del modello, eliminiamo il rischio di 'data leak age' 
e possiamo verificare che il modello non soffra di overfitting, su nessuno dei set.

Ma ci sono due problemi:
1- dovendo estrarre un ulteriore sub set dal data set, andiamo a ridurre ulteriormente il train set,
    il che è un grosso problema se stiamo lavorando con un data set già di per se abbastanza piccolo;
2- il modello potrebbe dipendere da come abbiamo eseguito lo split dei tre sub set. 

Esiste una soluzione migliore e si chiama 'K-Fold cross validation'.
Funziona così, prendiamo il dataset e lo dividiamo in un numero determinato di parti chiamato appunto Fold,
quello che dobbiamo fare è costruire 4 modelli con stessi iper parametri,
utilizzando ogni volta una delle 4 fold con test set e le restanti come train set. 

Calcoliamo l'errore per ciascun modello, ricorda che i modelli sono dello stesso tipo e con gli stessi iper parametri,
differiscono solo per il set di addestramento.
Calcoliamo l'errore complessivo come la media degli errori dei modelli.
La pratica migliore è utilizzare la cross validation al posto del set di validazione,
ma tenere sempre un test set a parte per la valutazione finale.
Quindi gli step da seguire sono i seguenti:
1 - Creare un set di addestramento e uno di test
2 - Selezionare tipo di modello ed iperparametri
3 - Addestrare il modello tramite cross validation usando il set di addestramento
4 - Eventualmente ripetere gli step 2 e 3 fino ad aver ottenuto buoni risultati, cioè un modello che ci dia una funzione di costo abbastanza bassa.
5 - Riaddestrare il modello sull'intero set di addestramento
6 - Testare il modello sul test set
7 - Se il risultato è buon abbiamo il modello, altrimenti torniamo allo step 2.

Come scegliere il numero di Folds da utilizzare:
per la maggior parte dei casi 10 va bene,
ma se un data set è molto grande utilizzare 5,
se il daset è piccolo utilizzare 20.
Un caso particolare è quello di un dataset davvero piccolo,
in tal caso possiamo utilizzare una tecnica chiamata 'leave one out cross validation',
cioè lasciane uno fuori, quidni il valore di k è pari al numero di esempi nel dataset,
quindi ad ogni ciclo della cross validation verrà utilizzato un unico esempio 
per il test set e tutti gli altri per l'addestramento. 




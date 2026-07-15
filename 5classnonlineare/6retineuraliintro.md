
## Introduzione alle Reti Neurali Artificiali (ANN)

Le Reti Neurali Artificiali sono modelli matematici ispirati al funzionamento del cervello umano, eccezionali nel trovare pattern complessi in enormi quantità di dati come immagini, testo e suoni. Entrare nel mondo del Deep Learning significa padroneggiare non solo la loro struttura, ma anche come "imparano" e come poter guidare questo apprendimento.

### 1. L'Anatomia di una Rete Neurale

Una rete neurale è composta da "nodi" (i neuroni artificiali) organizzati in diversi strati, o layers, sovrapposti:

* **Input Layer (Strato di Input):** È il punto di ingresso in cui ogni nodo rappresenta una singola proprietà del tuo dataset.
* **Hidden Layers (Strati Nascosti):** Sono gli strati intermedi dove avviene la vera magia computazionale. Una rete può avere un solo strato nascosto o dozzine, portando così al concetto di Deep Learning.
* **Output Layer (Strato di Output):** È il risultato finale che fornisce la predizione, come ad esempio le probabilità di appartenenza a classi diverse.

### 2. Il Cuore Matematico: Come Impara la Rete?

L'addestramento di una rete neurale avviene in cicli continui, composti da tre fasi principali:

* **Forward Propagation (Propagazione in avanti):** I dati viaggiano strato per strato all'interno della rete. In ogni neurone i segnali provenienti dallo strato precedente vengono moltiplicati per un peso, sommati a un bias e calcolati tramite la formula: 
$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$


.
* **Loss Function (Calcolo dell'errore):** La differenza calcolata tra la previsione finale della rete e il target reale genera l'errore del modello.
* **Backpropagation (Retropropagazione):** La rete viaggia a ritroso calcolando i gradienti dell'errore rispetto a ogni singolo peso. Utilizzando un algoritmo di ottimizzazione, la rete aggiorna questi pesi per minimizzare l'errore al ciclo successivo.

### 3. Le Funzioni di Attivazione

Il valore $z$ calcolato durante la propagazione in avanti viene sempre "filtrato" da una funzione di attivazione per introdurre la non-linearità. Senza queste funzioni, una rete neurale, per quanto profonda, si comporterebbe esattamente come un banale modello di Regressione Lineare. La non-linearità permette alla rete di piegare e curvare i confini di decisione, risolvendo problemi molto complessi.

Le funzioni più utilizzate includono:

* **ReLU (Rectified Linear Unit):** Restituisce 0 se il numero è negativo, altrimenti lo lascia invariato tramite la formula $f(x) = \max(0, x)$. È lo standard assoluto per gli strati nascosti (hidden layers) perché previene il blocco dell'apprendimento ed è computazionalmente leggerissima.
* **Leaky ReLU:** È una versione evoluta della ReLU creata per risolvere il problema della "morte dei neuroni". Permette a una minuscola quantità di informazione di passare per gli input negativi usando la formula $f(x) = \max(\alpha x, x)$.
* **Sigmoide (Sigmoid):** Schiaccia i numeri tra 0 e 1 usando $f(x) = \frac{1}{1 + e^{-x}}$. È usata quasi esclusivamente nello strato di output per le classificazioni binarie, poiché negli strati intermedi causa il problema del "Vanishing Gradient" dove la rete smette di imparare.
* **Tanh (Tangente Iperbolica):** Simile alla Sigmoide, ma schiaccia i numeri in un intervallo tra -1 e 1 tramite la formula $f(x) = \tanh(x)$.
* **Softmax:** Prende un gruppo di numeri e li converte in probabilità percentuali che sommate restituiscono esattamente 1.0 (il 100%). È utilizzata nell'output layer specificamente per le classificazioni multiclasse.

### 4. Gli Iperparametri

Mentre i parametri, ovvero pesi e bias, vengono imparati in completa autonomia dalla rete guardando i dati, gli iperparametri sono le "manopole" che l'utente deve impostare manualmente prima dell'inizio dell'addestramento. Definiscono le regole del gioco; perfino la scelta della funzione di attivazione e dei suoi valori interni, come la costante $\alpha$ nella Leaky ReLU, rientra tra gli iperparametri.

| Iperparametro | Funzionalità | Dettagli Pratici |
| --- | --- | --- |
| **Learning Rate** | Definisce la dimensione del passo con cui si aggiornano i pesi durante la retropropagazione. | È il parametro più critico: se è troppo alto salta la soluzione ottimale, se è basso impiega troppo tempo a imparare. |
| **Numero di Hidden Layers** | Determina la profondità strutturale della rete. | Più strati affrontano problemi astratti e complessi, ma alzano notevolmente il rischio di overfitting. |
| **Neuroni per Layer** | Determina la larghezza (la capacità) di un singolo strato. | Strati troppo larghi memorizzano i dati a memoria, strati troppo stretti causano underfitting. |
| **Epochs (Epoche)** | Indica il numero di volte in cui l'intera rete legge l'intero dataset. | Configurare troppe epoche spinge il modello verso l'overfitting. |
| **Batch Size** | Indica i campioni processati alla volta prima dell'aggiornamento dei pesi. | Invece di aggiornare i pesi su singoli dati o su tutto il dataset, agisce a pacchetti. |
| **Ottimizzatore** | L'algoritmo che aggiorna materialmente i pesi. | Adam è l'ottimizzatore moderno più utilizzato perché adatta dinamicamente il Learning Rate per convergere in fretta. |
| **Dropout Rate** | È una tecnica di regolarizzazione utilizzata per combattere l'overfitting. | Spegne casualmente una percentuale di neuroni (es. il 20%), costringendo la rete a generalizzare meglio. |


---

### Il Deep Learning: Oltre il Machine Learning Tradizionale

Il **Deep Learning** (Apprendimento Profondo) è un sottoinsieme specifico del Machine Learning che si basa esclusivamente sull'utilizzo di Reti Neurali Artificiali dotate di molti strati nascosti (da qui il termine "Deep", profondo).

Mentre nel Machine Learning tradizionale (come le SVM o gli Alberi Decisionali) è il programmatore a dover estrarre e selezionare manualmente le caratteristiche migliori dai dati (processo chiamato *Feature Engineering*), il Deep Learning fa tutto da solo attraverso una **gerarchia di astrazioni**:

* I primi strati della rete imparano a riconoscere pattern semplicissimi (es. in un'immagine riconoscono linee, bordi e contrasti di colore).
* Gli strati intermedi combinano queste linee per formare concetti più complessi (es. forme geometriche, un occhio, un orecchio).
* Gli strati finali uniscono questi concetti per prendere la decisione finale (es. riconoscere il volto di una persona specifica).

Questa capacità di estrazione automatica delle *feature* rende il Deep Learning lo strumento definitivo per gestire dati non strutturati ad altissima dimensionalità, come immagini in alta risoluzione, file audio o testi interi.

---

### Il Ciclo di Addestramento (Training Loop) Passo per Passo

L'addestramento di una Rete Neurale non è un evento singolo, ma un processo iterativo che si ripete migliaia di volte. Questo ciclo si chiama *Training Loop* e si compone di passaggi matematici ben definiti:

* **Fase 1: Inizializzazione dei Pesi (Initialization)**
Prima di iniziare, la rete non sa nulla. Tutti i pesi ($w$) e i bias ($b$) vengono impostati a valori casuali, solitamente molto vicini allo zero. Inizializzarli tutti a zero bloccherebbe l'apprendimento, quindi l'asimmetria casuale iniziale è fondamentale.
* **Fase 2: Forward Propagation (Propagazione in Avanti)**
Un "pacchetto" di dati (definito dal *Batch Size*) entra nello strato di input. I segnali attraversano la rete moltiplicandosi per i pesi, sommandosi ai bias e passando attraverso le funzioni di attivazione. Questo processo genera una previsione finale (spesso completamente errata al primo ciclo).
* **Fase 3: Calcolo dell'Errore (Loss Calculation)**
La previsione della rete viene confrontata con la "verità assoluta" (il target reale del dataset). Una funzione matematica, la *Loss Function* (es. *Mean Squared Error* per la regressione o *Cross-Entropy* per la classificazione), quantifica numericamente quanto la rete ha sbagliato. L'obiettivo è portare questo numero il più vicino possibile allo zero.
* **Fase 4: Backpropagation (Retropropagazione del Gradiente)**
È il cuore matematico dell'apprendimento. Utilizzando la regola della catena del calcolo differenziale, la rete viaggia a ritroso dall'output verso l'input. Calcola il **gradiente** (la derivata parziale) della funzione di errore rispetto a *ogni singolo peso* della rete. In parole povere, scopre di chi è stata la colpa dell'errore e di quanto deve essere modificato ogni peso per migliorare al giro successivo.
* **Fase 5: Ottimizzazione (Weight Update)**
Una volta calcolati i gradienti, interviene l'Ottimizzatore (es. *Adam* o *SGD*). L'ottimizzatore aggiorna fisicamente i valori dei pesi sottraendo una frazione del gradiente calcolato, moltiplicata per il *Learning Rate* ($\eta$). La formula base dell'aggiornamento è:

$$w_{nuovo} = w_{vecchio} - \eta \cdot \frac{\partial L}{\partial w}$$


* **Fase 6: Iterazione per Epoche**
I passaggi dal 2 al 5 vengono ripetuti per il batch successivo, finché l'intero dataset non è stato processato. Questo completa **una singola Epoca**. Il processo ricomincia da capo per il numero di epoche definite dai tuoi iperparametri, affinando i pesi a ogni passaggio e facendo calare progressivamente l'errore.
* **Fase 7: Validazione (Validation)**
Alla fine di ogni Epoca, la rete viene messa alla prova su un set di dati separato (il *Validation Set*) su cui non ha mai calcolato i gradienti e non ha mai aggiornato i pesi. Questo serve a monitorare l'overfitting: se l'errore sul Training Set continua a scendere, ma l'errore sul Validation Set inizia a salire, significa che la rete sta iniziando a imparare i dati a memoria. A questo punto, solitamente, si interrompe l'addestramento (*Early Stopping*).
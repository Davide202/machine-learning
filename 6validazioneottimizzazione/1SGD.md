
---

## Varianti del Gradient Descend

La versione che abbiamo visto fin ora è chiamata **Full Batch Gradient Descend**, in cui utilizziamo l'intero trainset ad ogni iterazione:

* Prendiamo tutti gli esempi del train set.
* Facciamo uno step del Gradient Descend che proverà ad ottimizzare i coefficienti del modello cioè i suoi pesi ed il bias, in modo da minimizzare la funzione di costo.
* Ripetiamo il processo utilizzando sempre tutti gli esempi del trainset insieme ed ottenendo ad ogni epoca dei coefficienti migliori, quindi un minor valore della funzione di costo.

Ripetendo questo processo per un numero sufficiente di epoche potremmo riuscire a trovare i coefficienti del modello che permettono di ottenere il valore più basso della funzione di costo e quindi il modello migliore.

**Questo processo ha dei limiti:**

* **Inefficiente per dataset grandi:** deve caricare l'intero dataset in memoria, quindi è molto dispendioso in termini di risorse di calcolo soprattutto per dataset molto grandi.
* **Poco dinamico:** per migliorare il modello con nuovi dati bisogna riaddestrarlo sull'intero dataset.
* **Soggetto al problema dei minimi locali:** si presenta quando la funzione di costo che vogliamo minimizzare non è convessa, cioè quando sono presenti più minimi locali. Problema comune quando lavoriamo in uno spazio dimensionale piuttosto alto, utilizzando il Full Batch Gradient Descend corriamo il rischio che questo possa incepparsi in un minimo locale piuttosto che raggiungere il minimo globale.

Una variante che risolve questi problemi è lo **Stochastic Gradient Descend**, in cui piuttosto che utilizzare l'intero trainset ad ogni iterazione del Gradient Descend utilizziamo un solo ed unico esempio:

* Selezioniamo un esempio dal trainset che utilizziamo per fare uno step del Gradient Descend, che tornerà dei coefficienti, quindi dei pesi e bias che permettono di ottenere un determinato valore della funzione di costo.
* Prendiamo un altro esempio ed eseguiamo un altro step del Gradient Descend che aggiornerà pesi e bias del modello e darà un altro valore per la funzione di costo.

Con questa tecnica il valore della funzione di costo tenderà a diminuire con andamento oscillatorio. Non confondere Step e Epoca, questi sono step del Gradient Descend, l'epoca si conclude dopo che il Gradient Descend è passato sull'intero dataset. Al termine di ogni epoca, questo va mischiato per evitare i cicli all'interno del Gradient Descend. Quindi anche per lo Stochastic, dopo un numero sufficiente di epoche riusciremo a minimizzare la funzione di costo e quindi ad ottenere i coefficienti del nostro modello.

**Lo Stochastic Gradient Descend ha quindi dei vantaggi:**

* **Pesa poco in memoria:** è sufficiente caricare un esempio per volta.
* **Molto dinamico:** per aggiornare il modello con nuovi dati è possibile eseguire uno step del SGD solo su questi dati.
* **Meno soggetto al problema dei minimi locali:** grazie alle fluttuazioni della funzione di costo potrebbe sfuggire ai minimi locali.

**Lo Stochastic Gradient Descend ha anche degli svantaggi:**

* **Fluttuazioni eccessive della funzione di costo:** rischia di saltare il punto di minimo globale.

Per ovviare a questi problemi esiste una soluzione, una via di mezzo, e si chiama **Mini Batch Gradient Descend**, che permette di eseguire il Gradient Descend su un numero determinato di esempi per volta.
Ad esempio prendendo un batch di 4 esempi, il Gradient Descend verrà eseguito su 4 esempi per volta. Anche qui la funzione di costo oscillerà ma non così tanto come nello Stochastic, quindi aumenterà le possibilità di raggiungere il punto di minimo globale per la funzione di costo. Il trucco sta nel trovare la dimensione corretta del mini batch del set di addestramento, che permette di bilanciare difetti e benefici del Full Batch e dello Stochastic. Questo valore, il batch-size è un altro dei nostri iperparametri.

---

### 💡 Precisazioni e Approfondimenti (Note Aggiuntive)

* **Online vs Offline Learning:** Quando nei tuoi appunti scrivi giustamente che il Full Batch è "poco dinamico" e lo Stochastic è "molto dinamico", nel gergo tecnico stai descrivendo la differenza tra l'*Offline Learning* (imparare tutto in blocco) e l'*Online Learning* (imparare un nuovo dato al volo non appena è disponibile).
* **Il trucco per scegliere il Batch-Size:** Nel Mini Batch, i valori ottimali per la dimensione dei pacchetti sono sempre **potenze di 2** (es. 32, 64, 128, 256). Il motivo non è puramente matematico, ma hardware: le memorie dei computer e delle GPU sono strutturate per elaborare pacchetti binari a una velocità nettamente superiore.
* **L'importanza del rimescolamento (Shuffling):** Nel paragrafo dello Stochastic hai colto un dettaglio cruciale: mischiare il dataset al termine di ogni epoca. Se non mescolassimo i dati, il modello imparerebbe a riconoscere l'ordine con cui glieli passiamo, invece di imparare il vero significato delle informazioni contenute.






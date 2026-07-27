Riduzione della Dimensionalità
Principal Component Analysis

Perchè ridurre la dimensionalità:
- ridurre la complessità computazionale, velocizzare il processo di apprendimento dei pesi,
    Meno proprietà = meno pesi da ottimizzare = meno tempo e risorse richieste per l'addestramento
- per poterlo esplorare e visualizzare graficamente

Quando si tratta di ridurre la dimensionalità del dataset abbiamo due possibilità:
- Feature Selection
    selezioniamo sperimentalmente un insieme di proprietà da utilizzare, scartando le proprietà meno utili
    e lasciando quelle che danno più informazione.
- Feature Extration
    estraiamo nuove proprietà dalle proprietà da quelle che già abbiamo,
    bisogna prima standardizzare le proprietà di partenza e portarle quindi sulla stessa scala
    e creiamo le nuove proprietà come combinazioni lineari delle proprietà del modello,
    andiamo quindi a studiare le nuove proprietà, che saranno meno delle nuove.
    Questa tecnica è la Principal Component Analysis, è la tecnica di riduzione della dimensionalità NON SUPERVISIONATA più utilizzata.

La Principal Component Analysis (PCA) è non supervisionata perchè non stiamo considerando la variabile target,
questo perchè esegue la riduzione dimensionale usando unicamente le features,
consiste nel comprimere un insieme di proprietà correlate tra di loro in un unica proprietà.,
lo fa con l'assunzione che le proprietà che contengono più informazione sono quelle che hanno una varianza maggiore.

La varianza misura la variabilità dei valori all'interno di una distribuzione.

La PCA ci permette di individuare la direzione di maggior varianza che viene chiamata First Principal Component (prima componente principale),
individuata questa è possibile identificare un numero di componenti principali aggiuntive pari al numero di dimensioni del nostro dataset.

Le componenti principali successive alla prima devono essere ortonali alla prima,
la limitazione dell'ortogonalità ci permette di trovare assi indipendenti tra di loro,
quindi contenenti soltanto informazioni differenti.

Come definire il numero di componenti principali?
- Definirlo manualmente ad esempio se il nostro scopo è visualizzare il dataset in 2 dimensioni dobbiamo tenere le prime due componenti principali.
- Scegliere un valore minimo di varianza che vogliamo mantenere dopo aver eseguito la riduzione dimensionale e selezionare il numero di componenti principali di conseguenza.






Linear Discriminant Analysis

Differenze tra PCA e LDA:

PCA: metodo non supervisionato, trova le componenti principali di massima varianza,
    senza considerare eventuali label nei dati
LDA: metodo supervisionato, quindi tiene conto delle label,
     trova il sottospazione dimensionale che massimizza la separabilità delle classi.

La LDA si preoccupa della varianza delle singole classi,
minimizza la varianza nelle classe e
massimizza la distanza tra i punti medi.

LDA identifica un numero di discriminanti pari al numero delle classi -1.

Utilizzare la PCA risolve i problemi non supervisionati e se le classi all'interno del dataset sono sbilanciate in numerosità degli esempi,
in tutti gli altri casi utilizzare LDA.

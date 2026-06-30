
# Perchè e come suddividere il dataset 
# suddividiamo il dataset in due set
# uno da fornire per l'addestramento (train set)
# uno da utilizzare per il test del modello (test set)

# Criterio da utilizzare per lo split:
# il train set deve essere molto più grande del test set,
# per esempio 70% e 30%.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Caricamento del dataset locale
housing_path = "../data/housing.csv"
boston_df = pd.read_csv(housing_path)

# Rinominiamo la colonna 'medv' in 'TARGET' per mantenere 
# la coerenza visiva con il tuo script originale
boston_df = boston_df.rename(columns={'medv': 'TARGET'})

# 2. Creazione di X e Y
# X contiene tutti i valori tranne la colonna target
X = boston_df.drop('TARGET', axis=1).values
# Y contiene esclusivamente i valori del target
Y = boston_df['TARGET'].values

print("X.shape:", X.shape) # Dovrebbe essere circa (506, 13)

# 3. Suddivisione automatica con Scikit-Learn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)

# Mostriamo le prime righe del dataset completo
boston_df.head()

# %%
# Per effettuare la suddivisione iniziamo dal test set (Metodo manuale con Pandas)

# Campiona casualmente il 30% delle righe per il test set
boston_test_df = boston_df.sample(frac=0.3)

# Crea il train set eliminando le righe appena pescate per il test
boston_train_df = boston_df.drop(boston_test_df.index)

print("Numero di esempi nel Train Set: " + str(boston_train_df.shape[0]))
print("Numero di esempi nel Test Set: " + str(boston_test_df.shape[0]))




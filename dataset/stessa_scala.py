## Portare i dati sulla stessa scala è importante perchè
# l'algoritmo di apprendimento potrebbe dare più importanza
# alle colonne con range di valori più grandi
# in base ai modelli che decidiamo di utilizzare

## Abbiamo due metodi per portare i dati sulla stessa scala:
# 1 - NORMALIZZAZIONE 
# (portare i dati in un range tra 0 e 1) 
# (x_i - x_min)/(x_max - x_min)
# 2 - STANDARDIZZAZIONE
# creare una distrubuzione centrata a 0 con deviazione standard 1
# (distribuzione normale)
# il range di valori sarà tra -1 e +1
# x_i_std = (x_i - x_mean)/x_sd 
# x_sd = deviazione standard

## Quale metodo utilizzare:
# Normalizzazione:
# - alcuni modelli non lineari come le reti neurali richiedono che i dati siano in un range tra 0 e 1
# - l'intensità dei pixel nelle immagini vanno normalizzati
# Standardizzazione:
# - Mantiene le informazioni riguardo gli outliers
# - in alcuni modelli lineari avere i dati in forma di distribuzione normale agevola la fase di addestramento


# %%
import pandas as pd
import numpy as np

wines_path = "../../data/wines.csv"

wines = pd.read_csv(wines_path,
                    usecols=[0,1,7],
                    names=['classe','alcol','flavonoidi'])
Y = wines['classe'].values
X = wines.drop('classe',axis=1).values

wines.head(10)

# %%
wines.describe()

# %%
# Applichiamo la normalizzazione manualmente

wines_norm = wines.copy()
features = ["alcol","flavonoidi"]
to_norm = wines[features]

wines_norm[features] = (to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
wines_norm.head()

# %%
# Applichiamo la normalizzazione con MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_norm = X.copy()
X_norm = mms.fit_transform(X_norm)
X_norm[:5]

# %%
# Applichiamo la standardizzazione manualmente

wines_std = wines.copy()
to_std = wines_std[features]
wines_std[features] = (to_std - to_std.mean())/(to_std.std())
wines_std[:5]

# %%
# Applichiamo la standardizzazione con StandardScaler

from sklearn.preprocessing import StandardScaler

X_std = X.copy()
ss = StandardScaler()

X_std = ss.fit_transform(X_std)

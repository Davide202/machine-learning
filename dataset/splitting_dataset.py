
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

from sklearn.datasets import load_boston
boston = load_boston()

# boston.DESCR

X = boston.data
Y = boston.target

from sklearn.model_selection import train_test_split

X.shape # (506, 13)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

X_train.shape #(354, 13)
X_test.shape #(152, 13)

boston_df = pd.DataFrame(data=np.c_[boston['data'], boston['target']],
                         columns=np.append(boston['feature_names'],'TARGET'))

boston_df.head()

# %%
# Per effettuare la suddivisione iniziamo dal test set

boston_test_df = boston_df.sample(frac=0.3)
boston_train_df = boston_df.drop(boston_test_df.index)

print("Numero di esempi nel Train Set: " + str(boston_train_df.shape[0]))
print("Numero di esempi nel Test Set: " + str(boston_test_df.shape[0]))






















# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


boston_path = "../data/housing.csv"
boston = pd.read_csv(
    boston_path,
    header=0,
    names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"]
)

boston.head()
boston.info()


# Effettuiamo una regressione lineare multipla sfruttando due di queste proprietà
# per scegliere queste proprietà possiamo guardare la correlazione tra le variabili:

boston.corr()

import seaborn as sns

sns.heatmap(boston.corr(),xticklabels=boston.columns,yticklabels=boston.columns)

cols = ["RM","LSTAT","PRATIO","TAX","INDUS","MEDV"]

sns.heatmap(
    boston[cols].corr(),
    xticklabels=boston[cols].columns,
    yticklabels=boston[cols].columns,
    annot=True,
    annot_kws={'size':12}

)


sns.pairplot(boston[cols])

# %%

X = boston[["RM","LSTAT"]].values
Y = boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

ll = LinearRegression()
ll.fit(X_train, Y_train)
Y_pred = ll.predict(X_test)

print("MSE : " + str(mean_squared_error(Y_test,Y_pred)))
print("R2 : " + str(r2_score(Y_test,Y_pred)))

# %%

X = boston.drop("MEDV",axis=1).values
Y = boston["MEDV"].values 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

ll = LinearRegression()

ll.fit(X_train_std,Y_train)
Y_pred = ll.predict(X_test_std)

print("MSE : " + str(mean_squared_error(Y_test,Y_pred)))
print("R2 : " + str(r2_score(Y_test,Y_pred)))

list(zip(boston.columns, ll.coef_))


# %%



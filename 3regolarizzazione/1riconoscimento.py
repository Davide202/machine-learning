

# %%

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

boston_path = "../data/housing.csv"
boston = pd.read_csv(
    boston_path,
    header=0,
    names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"]
)

X = boston.drop('MEDV',axis=1).values 
Y = boston['MEDV'].values 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

poly_feats = PolynomialFeatures(degree=2)
X_train_poly = poly_feats.fit_transform(X_train)
X_test_poly = poly_feats.fit_transform(X_test)

X_train_poly.shape #(354, 105)

# %%
# Prima di eseguire la regressione polinomiale, è necessario standardizzare i dati.
# Questo perché le caratteristiche polinomiali possono avere scale molto diverse, 
#e la standardizzazione aiuta a migliorare le prestazioni del modello.

ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)

ll = LinearRegression()
ll.fit(X_train_poly,Y_train)

Y_pred_train = ll.predict(X_train_poly)

mse_train = mean_squared_error(Y_train,Y_pred_train)
r2_train = r2_score(Y_train,Y_pred_train)

print(f"Train MSE: {mse_train:.2f}")
print(f"Train R2: {r2_train:.2f}")  

# %%

Y_pred_test = ll.predict(X_test_poly)

mse_test = mean_squared_error(Y_test,Y_pred_test)
r2_test = r2_score(Y_test,Y_pred_test)

print(f"Test MSE: {mse_test:.2f}")
print(f"Test R2: {r2_test:.2f}")    

# Sia MSE che R2 sono decisamente peggiori rispetto a quanto avviene sul set di addestramento. 
# Questo è un chiaro segnale di overfitting, 
# dove il modello si adatta troppo bene ai dati di addestramento e non generalizza bene sui dati di test.

# %%


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


boston_path = "../../data/housing.csv"
boston = pd.read_csv(
    boston_path,
    header=0,
    names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"]
)

# scegliere la proprietà

cols = ["RM","LSTAT","DIS","RAD","MEDV"]

sns.pairplot(boston[cols])

# %%

X = boston[["LSTAT"]].values
Y = boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


from sklearn.preprocessing import PolynomialFeatures

for i in range(1,11):
    polyfeats = PolynomialFeatures(degree=i)

    X_train_poly = polyfeats.fit_transform(X_train)
    X_test_poly = polyfeats.transform(X_test)

    X_train[:3]

    X_train_poly[:3]

    ll = LinearRegression()
    ll.fit(X_train_poly,Y_train)
    Y_pred = ll.predict(X_test_poly)

    mes = mean_squared_error(Y_test,Y_pred)
    r2 = r2_score(Y_test,Y_pred)

    print("DEGREE: " + str(i) + " - MSE: " + str(mes) + " - R2: " + str(r2))
    # Dalla stampa di questi risultati è possibile osservare il fenomeno di OVERFITTING 



# %%
# sfruttiamo tutte le proprietà del modello

X = boston.drop("MEDV",axis=1).values 
Y = boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


from sklearn.preprocessing import PolynomialFeatures

for i in range(1,5):
    polyfeats = PolynomialFeatures(degree=i)

    X_train_poly = polyfeats.fit_transform(X_train)
    X_test_poly = polyfeats.transform(X_test)

    X_train[:3]

    X_train_poly[:3]

    ll = LinearRegression()
    ll.fit(X_train_poly,Y_train)
    Y_pred = ll.predict(X_test_poly)

    mes = mean_squared_error(Y_test,Y_pred)
    r2 = r2_score(Y_test,Y_pred)

    print("DEGREE: " + str(i) + " - MSE: " + str(mes) + " - R2: " + str(r2))
    # Dalla stampa di questi risultati è possibile osservare il fenomeno di OVERFITTING 

     




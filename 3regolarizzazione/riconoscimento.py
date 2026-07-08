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
# Come contrastare l'overfitting con la regolarizzazione L1 e L2

# La regolarizzazione penalizza i pesi più grandi, riducendo la complessità del modello e migliorando la sua capacità di generalizzazione.
# Riduce la varianza del modello, ma può aumentare il bias. 
# L'obiettivo è trovare un equilibrio tra bias e varianza per ottenere un modello che generalizzi bene sui dati non visti.

# Entrambe le regolarizzazioni L1 e L2 aggiungono un termine di penalizzazione alla funzione di costo della regressione lineare, 
# ma lo fanno in modi diversi:
# - L1 (Lasso): aggiunge la somma dei valori assoluti dei pesi al termine di costo. 
#             Questo può portare a modelli più semplici, poiché alcuni pesi possono diventare esattamente zero, 
#             eliminando alcune caratteristiche.
# - L2 (Ridge): aggiunge la somma dei quadrati dei pesi al termine di costo. 
#             Questo tende a ridurre i pesi, ma raramente li porta a zero, 
#             quindi tutte le caratteristiche rimangono nel modello.    


# Come scegliere il valore del parametro di regolarizzazione (alpha):
# Il parametro di regolarizzazione (alpha) controlla la forza della penalizzazione.
# Un valore più alto di alpha aumenta la penalizzazione, riducendo i pesi e semplificando il modello,
# ma può anche aumentare il bias. 
# Un valore più basso di alpha riduce la penalizzazione, 
# permettendo al modello di adattarsi meglio ai dati di addestramento, ma può aumentare la varianza e il rischio di overfitting. 
# Una soluzione corretta è quella di utilizzare la validazione incrociata per trovare il valore ottimale di alpha che bilancia bias e varianza,
# ottenendo così un modello che generalizza bene sui dati non visti.
# Cercare in uno spazio di potenze di 10, ad esempio da 10^-4 a 10^4 (anche 10^1), 
# può essere un buon punto di partenza per trovare il valore ottimale di alpha. 

# Ridge, Lasso ed ElasticNet

# %%
import pandas as pd
# import numpy as np
# from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

boston_path = "../data/housing.csv"
boston = pd.read_csv(
    boston_path,
    header=0,
    names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"]
)

X = boston.drop('MEDV', axis=1).values 
Y = boston['MEDV'].values 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

poly_feats = PolynomialFeatures(degree=2)
X_train_poly = poly_feats.fit_transform(X_train)
X_test_poly = poly_feats.transform(X_test)

print("Forma di X_train_poly:", X_train_poly.shape)

# %%
# Prima di eseguire la regressione polinomiale, è necessario standardizzare i dati.
# Questo perché le caratteristiche polinomiali possono avere scale molto diverse, 
# e la standardizzazione aiuta a migliorare le prestazioni del modello.

ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)


# %%
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Un modello di Machine Learning che applica la regolarizzazione L2 è Ridge Regression.

from sklearn.linear_model import Ridge

print("\n--- RIDGE REGRESSION (L2) ---")

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, Y_train)

    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)

    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)

    print(f"Alpha: {alpha:7} | Train MSE: {mse_train:5.2f}, Train R2: {r2_train:5.2f} | Test MSE: {mse_test:5.2f}, Test R2: {r2_test:5.2f}")


# Otteniamo un valore ottimale di alpha che bilancia bias e varianza, 
# ottenendo il miglior risultato per un valore di alpha = 10. 
# 
# %%
# Un modello di Machine Learning che applica la regolarizzazione L1 è Lasso Regression.

from sklearn.linear_model import Lasso

print("\n--- LASSO REGRESSION (L1) ---")

for alpha in alphas:
    # Aggiunto max_iter=10000 per garantire la convergenza
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, Y_train)

    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)

    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)

    print(f"Alpha: {alpha:7} | Train MSE: {mse_train:5.2f}, Train R2: {r2_train:5.2f} | Test MSE: {mse_test:5.2f}, Test R2: {r2_test:5.2f}")

# La regolarizzazione è stata applicata con un punteggio migliore 
# per un valore di alpha = 0.1, che ha portato a un modello più semplice con meno caratteristiche,
# ma con una buona capacità di generalizzazione sui dati di test.

# Per alpha = 10, l'effetto della regolarizzazione è stato troppo forte, 
# i pesi sono stati ridotti troppo, portando a un modello con un alto bias e una bassa capacità di generalizzazione.


# %%
# Un modello di Machine Learning che applica sia la regolarizzazione L1 che L2 è ElasticNet Regression.

from sklearn.linear_model import ElasticNet

print("\n--- ELASTICNET REGRESSION (L1 + L2) ---")

for alpha in alphas:
    # Aggiunto max_iter=10000 per garantire la convergenza
    model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)  # l1_ratio=0.5 indica un equilibrio tra L1 e L2  
    model.fit(X_train_poly, Y_train)

    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)

    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)

    print(f"Alpha: {alpha:7} | Train MSE: {mse_train:5.2f}, Train R2: {r2_train:5.2f} | Test MSE: {mse_test:5.2f}, Test R2: {r2_test:5.2f}")

# Per un alpha = 0.01 otteniamo sia miglio punteggio che minor error e l'overfitting è stato ridotto, 
# con un modello più semplice e una buona capacità di generalizzazione sui dati di test.  
# 
# Ultima cosa da tenere a mente è che quando utilizzi la regolarizzazione 
# i dati devono essere sulla stessa scala, qundi è necessario utilizzare normalizzazione o standardizzazione prima di applicare la regolarizzazione.    

# %%


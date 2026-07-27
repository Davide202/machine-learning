# Linear Discriminant Analysis


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression



iris_path = "../data/iris.csv"
iris = pd.read_csv(
    iris_path, 
    header=0, 
    names=["sepal length","sepal width","petal length","petal width","target"]
)

# Mostra le PRIME 10 righe (Senza print!)
iris.head(10)

X = iris.drop("target",axis=1).values 
Y = iris["target"].values 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=0)


le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.decomposition import PCA 

pca = PCA(n_components=2)

pc_train = pca.fit_transform(X_train)
pc_test = pca.transform(X_test)

plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.scatter(pc_train[:,0],pc_train[:,1], c=Y_train)
plt.scatter(pc_test[:,0],pc_test[:,1], c=Y_test, alpha=0.5)

# %% 

lr = LogisticRegression()
lr.fit(pc_train,Y_train)

accuracy_train = accuracy_score(Y_train, lr.predict(pc_train))
accuracy_test = accuracy_score(Y_test, lr.predict(pc_test))

loss_train = log_loss(Y_train, lr.predict_proba(pc_train))
loss_test = log_loss(Y_test, lr.predict_proba(pc_test))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))



# %%
# Ripetiamo il processo con LDA
# prima di farlo ci dobbiamo assicurare che le classi siano ben bilanciate


np.unique(Y,return_counts=True) # ci sono 50 esempi per ogni classe, sono perfettamente bilanciate

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 

lda = LDA(n_components=2)

ld_train = lda.fit_transform(X_train,Y_train)
ld_test = lda.transform(X_test)

plt.xlabel("Primo discriminante")
plt.ylabel("Secondo discriminante")
plt.scatter(ld_train[:,0],ld_train[:,1], c=Y_train)
plt.scatter(ld_test[:,0],ld_test[:,1], c=Y_test, alpha=0.5)


lr = LogisticRegression()
lr.fit(ld_train,Y_train)

accuracy_train = accuracy_score(Y_train, lr.predict(ld_train))
accuracy_test = accuracy_score(Y_test, lr.predict(ld_test))

loss_train = log_loss(Y_train, lr.predict_proba(ld_train))
loss_test = log_loss(Y_test, lr.predict_proba(ld_test))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))



# %%

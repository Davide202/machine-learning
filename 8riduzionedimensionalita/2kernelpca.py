# Vediamo come utilizzare la PCA insieme ad una funzione Kernel
# per passare da uno spazio dimensionale non linere
# cioè le proprietà sono legate da proprietà non lineari,
# in uno spazio lineare con dimensione minore.


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss
from time import time

X,Y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=1)

plt.scatter(X[:,0],X[:,1], c=Y)

X_train ,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

lr = LogisticRegression()
start_time = time()
lr.fit(X_train,Y_train)
end_time = time()

print("TEMPO DI ADDESTRAMENTO: " + str(end_time - start_time))

accuracy_train = accuracy_score(Y_train, lr.predict(X_train))
accuracy_test = accuracy_score(Y_test, lr.predict(X_test))

loss_train = log_loss(Y_train, lr.predict_proba(X_train))
loss_test = log_loss(Y_test, lr.predict_proba(X_test))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))

# l'accuracy in entrambi i casi è intorno allo 0.5 quindi è inutile

from viz import plot_boundary

plot_boundary(lr,X,Y)

# %%

from sklearn.decomposition import KernelPCA 

kpca = KernelPCA(kernel='rbf', gamma=5)

kpc = kpca.fit_transform(X)

plt.scatter(kpc[:,0],kpc[:,1],c=Y)

# %%

plt.scatter(kpc[:,0],np.zeros((1000,1)),c=Y)

fcp = kpc[:,0]
fcp = fcp.reshape(-1,1)
fcp.shape
lr = LogisticRegression()

X_train ,X_test, Y_train, Y_test = train_test_split(fcp,Y,test_size=0.2,random_state=1)

lr = LogisticRegression()
start_time = time()
lr.fit(X_train,Y_train)
end_time = time()

print("TEMPO DI ADDESTRAMENTO: " + str(end_time - start_time))

accuracy_train = accuracy_score(Y_train, lr.predict(X_train))
accuracy_test = accuracy_score(Y_test, lr.predict(X_test))

loss_train = log_loss(Y_train, lr.predict_proba(X_train))
loss_test = log_loss(Y_test, lr.predict_proba(X_test))

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))


# %%

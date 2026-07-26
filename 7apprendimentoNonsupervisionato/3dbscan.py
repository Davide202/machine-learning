

# %%

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

plt.rcParams["figure.figsize"] = (14, 10)
sns.set_theme()

from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.5, random_state= 0)

plt.scatter(X[:,0],X[:,1],s=50)


# %%
# Proviamo il K-Means

from sklearn.cluster import KMeans

km = KMeans(n_clusters=2)

y_km = km.fit_predict(X)

plt.scatter(X[:,0],X[:,1],s=50, c=y_km, cmap="viridis")



# %%
# Proviamo il clustering gerarchico agglomerativo

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=2, linkage='ward')

y_hc = hc.fit_predict(X)

plt.scatter(X[:,0],X[:,1],s=50, c=y_hc, cmap="viridis")


# %% 
# Proviamo con il DBSCAN

from sklearn.cluster import DBSCAN 

dbscan = DBSCAN(eps=0.25, min_samples=3) # il nostro grafico ha 2 dimensioni quindi proviamo con 3

y_dbscan = dbscan.fit_predict(X)

plt.scatter(X[:,0],X[:,1],s=50, c=y_dbscan, cmap="viridis")
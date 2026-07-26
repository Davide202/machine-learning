# Cluster Genarchico Agglomerativo 

# %%
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

plt.rcParams["figure.figsize"] = (14, 10)
sns.set_theme()

from sklearn.datasets._samples_generator import make_blobs 

X, _ = make_blobs(n_samples=100, centers=2, cluster_std= .5, random_state= 0)

plt.scatter(X[:,0],X[:,1],s=50)


# %%
# Costruire il dendrogramma

from scipy.cluster.hierarchy import linkage, dendrogram

# Costruiamo la matrice di associazione
link_matrix = linkage(X,method='ward')

# Visualizzazione tabellare della matrice di associazione
pd.DataFrame(link_matrix)

# Visualizzazione del Dendrogramma 
dendrogram(link_matrix)



# %%

# Dobbiamo scegliere una soglia con la quale scegliere i cluster sottostante
# Guardando il grafico la soglia si trova intorno a 10 cioè un numero di cluster pari a 3

from sklearn.cluster import AgglomerativeClustering 

hc = AgglomerativeClustering(n_clusters=3)

y = hc.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=y, s=200, cmap="viridis", edgecolors="black")


# %%

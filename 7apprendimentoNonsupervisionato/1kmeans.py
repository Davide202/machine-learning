

# %%
import matplotlib.pyplot as plt
import seaborn as sns 
plt.rcParams["figure.figsize"] = (12, 8)
# sns.set() # The function "set" is deprecated. Function `set` is deprecated in favor of `set_theme`
sns.set_theme()


from sklearn.datasets._samples_generator import make_blobs 

X, _ = make_blobs(n_samples=500, centers=4, cluster_std= .8, random_state= 0)

plt.scatter(X[:,0],X[:,1],s=50)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)
km.fit(X)


# %%
# stiamo colorando il grafico con le classi predette da k-means

y = km.predict(X)

print(str(y[:10]))


plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap="viridis")


# Disegnamo i centroidi

centroids = km.cluster_centers_

plt.scatter(centroids[:,0], centroids[:,1], c="red", alpha=0.5, s=200)

# %%
# BONUS: Implementazione dell'ELBOW METHOD
# Calcoliamo l'SSD (Inertia) per diversi valori di K (da 1 a 10)

ssd = []
K_range = range(1, 11)

for k in K_range:
    km_test = KMeans(n_clusters=k, n_init='auto', random_state=0)
    km_test.fit(X)
    # L'attributo inertia_ contiene esattamente l'SSD!
    ssd.append(km_test.inertia_)

# Disegniamo il grafico del gomito
plt.figure(figsize=(10, 6))
plt.plot(K_range, ssd, marker='o', linestyle='--', color='b')
plt.title('Metodo del Gomito (Elbow Method)')
plt.xlabel('Numero di Cluster (K)')
plt.ylabel('SSD (Inertia)')
plt.xticks(K_range)
plt.show()

# Dal grafico sarà evidente che il "gomito" si piega in corrispondenza di K=4!
# %%



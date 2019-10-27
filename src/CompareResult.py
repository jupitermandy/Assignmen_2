from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from Dataset import loadCSV
from dunnIndex import dunn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = loadCSV()
algorithms = ['DBSCAN', 'EM', 'K-Means']
iEps = 36
minPts = 11
n_clusters = 3
time_list = {}
dunnIndexList = {}
silhouetteList = {}

# DBSCAN
print('DBSCAN : The number of cluster: %d' % n_clusters)
start_time_0 = time.time()
DBSCAN_Res = DBSCAN(eps=iEps, min_samples=minPts).fit(X)
end_time_0 = time.time()
time_list[0] = (end_time_0 - start_time_0) * 1000
print('DBSCAN : Run time: %s' % time_list[0])

core_samples_mask = np.zeros_like(DBSCAN_Res.labels_, dtype=bool)
core_samples_mask[DBSCAN_Res.core_sample_indices_] = True
db_cluster_labels = DBSCAN_Res.labels_
# Calculate dunn index
db_pred = pd.DataFrame(db_cluster_labels)
db_pred.columns = ['Species']

# we merge this dataframe with df
db_prediction = pd.concat([X, db_pred], axis=1)
db_k_list = []
for i in range(n_clusters):
    db_clus = db_prediction.loc[db_prediction.Species == i]
    db_k_list.append(db_clus.values)
dunnIndex = dunn(db_k_list)
dunnIndexList[0] = dunnIndex
print('DBSCAN: The value of Dunn Index: %s' % dunnIndex)

silhouette_avg = silhouette_score(X, db_cluster_labels)
silhouetteList[0] = silhouette_avg
print('DBSCAN : The average of Silhouette Score: %s' % silhouetteList[0])

# EM
print('EM : The value of eps : %d and minPts : %d' % (iEps, minPts))
start_time_1 = time.time()
EM_Res = GaussianMixture(n_components=n_clusters).fit(X)
end_time_1 = time.time()
time_list[1] = (end_time_1 - start_time_1) * 1000
print('EM : Run time: %s' % time_list[1])
em_cluster_labels = EM_Res.predict(X)

# Calculate dunn index
em_pred = pd.DataFrame(em_cluster_labels)
em_pred.columns = ['Species']

# we merge this dataframe with df
em_prediction = pd.concat([X, em_pred], axis=1)
em_k_list = []
for i in range(n_clusters):
    em_clus = em_prediction.loc[em_prediction.Species == i]
    em_k_list.append(em_clus.values)
dunnIndexList[1] = dunn(em_k_list)
print('EM : The value of Dunn Index: %s' % dunnIndexList[1])

silhouetteList[1] = silhouette_score(X, em_cluster_labels)
print('EM : The average of Silhouette Score: %s' % silhouetteList[1])

# K-Means
print('K-Means : The number of cluster: %d' % n_clusters)
start_time_2 = time.time()
KMeans_Res = KMeans(n_clusters=n_clusters)
end_time_2 = time.time()
time_list[2] = (end_time_2 - start_time_2) * 1000
print('K-Means : Run time: %s' % time_list[2])
km_cluster_labels = KMeans_Res.fit_predict(X)

# Calculate dunn index
km_pred = pd.DataFrame(km_cluster_labels)
km_pred.columns = ['Species']

# we merge this dataframe with df
km_prediction = pd.concat([X, km_pred], axis=1)
km_k_list = []
for i in range(n_clusters):
    km_clus = km_prediction.loc[km_prediction.Species == i]
    km_k_list.append(km_clus.values)
dunnIndexList[2] = dunn(em_k_list)
print('K-Means : The value of Dunn Index: %s' % dunnIndexList[2])

silhouetteList[2] = silhouette_score(X, km_cluster_labels)
print('K-Means : The average of Silhouette Score: %s' % silhouetteList[2])

# views
plt.figure()
plt.plot(list(algorithms), list(silhouetteList.values()), color='#aec7e8')
plt.ylabel("Average of Silhouette Score")
plt.xlabel("Algorithms")
plt.title("Silhouette Scores of Each Algorithms")
plt.grid(True)

plt.figure()
plt.plot(list(algorithms), list(time_list.values()), color='#2ca02c')
plt.ylabel("Run time (milliseconds)")
plt.xlabel("Algorithms")
plt.title("Run time of Each Algorithms")
plt.grid(True)

plt.figure()
plt.plot(list(algorithms), list(dunnIndexList.values()), color='#ff9896')
plt.ylabel("The value of Dunn Index")
plt.xlabel("Algorithms")
plt.title("The values of Dunn Index of Each Algorithms")
plt.grid(True)

plt.show()
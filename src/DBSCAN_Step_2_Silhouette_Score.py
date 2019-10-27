from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from dunnIndex import dunn
from Dataset import loadCSV
import time
import matplotlib.pyplot as plt

X = loadCSV()

min_samples_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
eps_list = [20, 25, 30, 30, 35, 35, 35, 36, 36, 36]

x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

count = 10
noise = {}
clusters = {}
dunnIndexScore = {}
silhouetteScore = {}
print("Parameters of DBSCAN: metric=euclidean")
for mI in range(count):
    print("========================")
    print("The value of minPts: %d" % min_samples_list[mI])
    print("The value of eps: %s" % eps_list[mI])
    start_time = time.time()
    db = DBSCAN(eps=eps_list[mI], min_samples=min_samples_list[mI], metric='euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    cluster_labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_ = list(cluster_labels).count(-1)
    clusters[mI] = n_clusters
    noise[mI] = n_noise_

    print("Estimated number of cluster: %d" % n_clusters)
    print("Estimated number of noise: %d" % n_noise_)

    # Calculate dunn index
    pred = pd.DataFrame(cluster_labels)
    pred.columns = ['Species']

    # we merge this dataframe with df
    prediction = pd.concat([X, pred], axis=1)
    k_list = []
    for i in range(n_clusters):
        clus = prediction.loc[prediction.Species == i]
        k_list.append(clus.values)
    dunnIndex = dunn(k_list)
    dunnIndexScore[mI] = dunnIndex
    print('The value of Dunn Index: %s' % dunnIndex)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouetteScore[mI] = silhouette_avg
    print("The average silhouette_score is :", silhouette_avg)


# visualization
# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b']


fig, axs = plt.subplots(1, 4)
fig.set_size_inches(11, 5)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 50, 1)
minor_ticks = np.arange(0, 50, 1)

# axs[0].set_title("The number of noise when metric=euclidean")
axs[0].set_yticks(major_ticks)
axs[0].set_yticks(minor_ticks, minor=True)

# And a corresponding grid
axs[0].grid(which='both')

# Or if you want different settings for the grids:
axs[0].grid(which='minor', alpha=0.2,)
axs[0].grid(which='major', alpha=0.5)
axs[0].plot(list(x_values), list(noise.values()), color='#1f77b4')
axs[0].set_xlabel("Index of pairs of eps and minPts")
axs[0].set_ylabel("Number of noise")

# axs[1].set_title("The Average of Silhouette Score")
axs[1].plot(list(x_values), list(silhouetteScore.values()), color='#aec7e8')
axs[1].set_ylabel("Average of Silhouette Score")
axs[1].set_xlabel("Index of pairs of eps and minPts")
axs[1].grid(True)


# axs[2].set_title("The values of Dunn Index")
axs[2].plot(list(x_values), list(dunnIndexScore.values()), color='#ff7f0e')
axs[2].set_ylabel("Value of Dunn Index")
axs[2].set_xlabel("Index of pairs of eps and minPts")
axs[2].grid(True)


# axs[3].set_title("The number of clusters when metric=euclidean")
axs[3].set_yticks(major_ticks)
axs[3].set_yticks(minor_ticks, minor=True)

# And a corresponding grid
axs[3].grid(which='both')

# Or if you want different settings for the grids:
axs[3].grid(which='minor', alpha=0.2,)
axs[3].grid(which='major', alpha=0.5)
axs[3].plot(list(x_values), list(clusters.values()), color='#ffbb78')
axs[3].set_ylabel("Number of clusters")
axs[3].set_xlabel("Index of pairs of eps and minPts")
# plt.title('Noise, Silhouette Score, Dunn Index and number of clusters with different pairs of eps with minPts', loc='center', fontsize=14, fontweight='bold')
# plt.suptitle('Noise, Silhouette Score, Dunn Index and number of clusters with different pairs of eps with minPts',
#              fontsize=14, fontweight='bold', verticalalignment='top')
plt.grid(True)
plt.tight_layout()
plt.show()

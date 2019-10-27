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
eps_list = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

col = 11
row = 10
noise = [[0] * col for _ in range(row)]
clusters = [[0] * col for _ in range(row)]
times = [[0] * col for _ in range(row)]
dunnIndexScore = [[0] * col for _ in range(row)]
silhouetteScore = [[0] * col for _ in range(row)]
# print("Parameters of DBSCAN: eps=36 metric=manhattan")
print("Parameters of DBSCAN: metric=euclidean")
minPI = 0
for min_samp in min_samples_list:
    epsI = 0
    for eps_chose in eps_list:
        print("========================")
        print("The value of minPts: %d" % min_samp)
        print("The value of eps: %s" % eps_chose)
        start_time = time.time()
        db = DBSCAN(eps=eps_chose, min_samples=min_samp, metric='euclidean').fit(X)
        end_time = time.time()
        times[minPI][epsI] = (end_time - start_time) * 1000
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        cluster_labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise_ = list(cluster_labels).count(-1)
        # clusters[min_samp] = n_clusters
        # noise[min_samp] = n_noise_
        clusters[minPI][epsI] = n_clusters
        noise[minPI][epsI] = n_noise_

        # print("Estimated number of cluster: %d" % n_clusters)
        print("Estimated number of noise: %d" % n_noise_)
        # print("Estimate run time is %s" % (end_time - start_time))

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
        dunnIndexScore[minPI][epsI] = dunnIndex
        print('The dunn index score is %s' % dunnIndex)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        # silhouette_avg = silhouette_score(X, cluster_labels)
        # silhouetteScore[minPI][epsI] = silhouette_avg
        # print("The average silhouette_score is :", silhouette_avg)
        epsI = epsI + 1
    minPI = minPI + 1

# visualization
# These are the colors that will be used in the plot
color_sequence = ['r','#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b']

# plt.figure()
# plt.title("The number of noise when metric=euclidean")
# for i in range(row):
#     plt.plot(list(eps_list), list(noise[i]), color=color_sequence[i])
# plt.legend(('minPts=2', 'minPts=3',
#             'minPts=4', 'minPts=5', 'minPts=6',
#             'minPts=7', 'minPts=8', 'minPts=9',
#             'minPts=10', 'minPts=11'),
#            loc='upper right')
# plt.xlabel("Value of eps")
# plt.ylabel("Number of noise")

# plt.figure()
# plt.title("The Average of Silhouette Score")
# for i in range(row):
#     plt.plot(list(eps_list), list(silhouetteScore[i]), color=color_sequence[i])
# plt.legend(('minPts=2', 'minPts=3',
#             'minPts=4', 'minPts=5', 'minPts=6',
#             'minPts=7', 'minPts=8', 'minPts=9',
#             'minPts=10', 'minPts=11'),
#            loc='upper right')
# plt.xlabel("Value of eps")
# plt.ylabel("Average of Silhouette Score")

print(dunnIndexScore)
plt.figure()
plt.title("The values of Dunn Index")
for i in range(row):
    plt.plot(list(eps_list), list(dunnIndexScore[i]), color=color_sequence[i])
plt.legend(('minPts=2', 'minPts=3',
            'minPts=4', 'minPts=5', 'minPts=6',
            'minPts=7', 'minPts=8', 'minPts=9',
            'minPts=10', 'minPts=11'),
           loc='upper right')
plt.xlabel("Value of eps")
plt.ylabel("Value of Dunn Index")

plt.figure()
plt.title("The number of clusters when metric=euclidean")
for j in range(row):
    plt.plot(list(eps_list), list(clusters[j]), color=color_sequence[j])
plt.legend(('minPts=2', 'minPts=3',
            'minPts=4', 'minPts=5', 'minPts=6',
            'minPts=7', 'minPts=8', 'minPts=9',
            'minPts=10', 'minPts=11'),
           loc='upper right')
plt.xlabel("Value of eps")
plt.ylabel("Number of clusters")

plt.figure()
plt.title("The run time when metric=euclidean")
for j in range(row):
    plt.plot(list(eps_list), list(times[j]), color=color_sequence[j])
plt.legend(('minPts=2', 'minPts=3',
            'minPts=4', 'minPts=5', 'minPts=6',
            'minPts=7', 'minPts=8', 'minPts=9',
            'minPts=10', 'minPts=11'),
           loc='upper right')
plt.xlabel("Value of eps")
plt.ylabel("Run time (milliseconds)")

plt.show()

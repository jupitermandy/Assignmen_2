from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dunnIndex import dunn
import pandas as pd
from Dataset import loadCSV
import time
import matplotlib.pyplot as plt

X = loadCSV()


range_n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
sse = {}
dunnIndex = {}
estimatedTime = {}
silhouetteScore = {}
clusters = {}
col = 30
row = 12

initialMethods = ["k-means++", "random"]
sseList = [[0] * col for _ in range(row)]
times = [[0] * col for _ in range(row)]
randomStates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# print("Parameters of K-MEANS: init=k-means++")
# for randomState in range(1):
iMethod = 0
for initialM in initialMethods:
    iClusterI = 0
    for n_clusters in range_n_clusters:
        print("====================================")
        print("Estimated number of cluster: %d" % n_clusters)
        # print("Value of random-state: %d" % randomState)
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        print(initialM)
        start_time = time.time()
        clusterer = KMeans(n_clusters=n_clusters, init=initialM)
        end_time = time.time()
        # estimatedTime[n_clusters] = (end_time - start_time) * 1000
        times[iMethod][iClusterI] = (end_time - start_time) * 1000
        # print("Estimated run time is %s" % (end_time - start_time))
        cluster_labels = clusterer.fit_predict(X)

        # Record sse value
        # sse[n_clusters] = clusterer.inertia_
        sseList[iMethod][iClusterI] = clusterer.inertia_
        print("Estimated SSE value is %s" % clusterer.inertia_)
        iClusterI = iClusterI + 1

        # Calculate Dunn Index
        # Store the K-means results in a dataframe
        # pred = pd.DataFrame(cluster_labels)
        # pred.columns = ['Species']
        #
        # # Merge this dataframe with df
        # prediction = pd.concat([X, pred], axis=1)
        # k_list = []
        # for i in range(n_clusters):
        #     clus = prediction.loc[prediction.Species == i]
        #     k_list.append(clus.values)
        # dunnIndex[n_clusters] = dunn(k_list)
        # print("The dunn index score is :", dunnIndex[n_clusters])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        # silhouette_avg = silhouette_score(X, cluster_labels)
        # silhouetteScore[n_clusters] = silhouette_avg
        # print("The average silhouette_score is :", silhouette_avg)
        # print("The number of iteration run is :", clusterer.n_iter_)
    iMethod = iMethod + 1

# Elbow diagram
color_sequence = ['#1f77b4', '#9467bd', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b']
plt.figure()
plt.title("SSE Values")
for i in range(2):
    plt.plot(list(range_n_clusters), list(sseList[i]), color=color_sequence[i])
plt.legend(('init=k-means++', 'init=random'),
            loc='upper right')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")

# plt.figure()
# plt.title("Initialization method - k-means++")
# for i in randomStates:
#     plt.plot(list(range_n_clusters), list(times[i]), color=color_sequence[i])
# plt.legend(('random-state=0', 'random-state=1', 'random-state=2',
#             'random-state=3', 'random-state=4', 'random-state=5',
#             'random-state=6', 'random-state=7', 'random-state=8',
#             'random-state=9', 'random-state=10', 'random-state=11'),
#             loc='upper right')
# plt.xlabel("Number of cluster")
# plt.ylabel("Run time (milliseconds)")

plt.show()
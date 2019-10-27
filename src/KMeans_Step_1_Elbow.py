from sklearn.cluster import KMeans
from Dataset import loadCSV
import matplotlib.pyplot as plt

X = loadCSV()

range_n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
sse = {}
clusters = {}
col = 30
row = 12
initialMethods = ["k-means++", "random"]
sseList = [[0] * col for _ in range(row)]
times = [[0] * col for _ in range(row)]
iMethod = 0
for initialM in initialMethods:
    iClusterI = 0
    for n_clusters in range_n_clusters:
        print("====================================")
        # print("Estimated number of cluster: %d" % n_clusters)
        clusterer = KMeans(n_clusters=n_clusters, init=initialM)
        cluster_labels = clusterer.fit_predict(X)

        # Record sse value
        sseList[iMethod][iClusterI] = clusterer.inertia_
        print("Estimated SSE value is %s when the number of cluster is %d initial method is %s" % (clusterer.inertia_, n_clusters, initialM))
        iClusterI = iClusterI + 1
    iMethod = iMethod + 1

# Elbow diagram
color_sequence = ['#1f77b4', '#9467bd', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b']
plt.figure()
plt.title("SSE Values")
for i in range(2):
    plt.plot(list(range_n_clusters), list(sseList[i]), color=color_sequence[i])
plt.axvline(x=3, color='#d62728')
plt.legend(('init=k-means++', 'init=random'),
            loc='upper right')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")

plt.show()
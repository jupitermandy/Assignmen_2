from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Dataset import loadCSV

X = loadCSV()

range_n_clusters = [3, 4, 5]
dunnIndex = {}
silhouetteScore = {}
mI = 0
print("Silhouette Scores of EM when n_clusters is [3, 4, 5]")
for n_clusters in range_n_clusters:
    print("Estimated number of cluster: %d" % n_clusters)
    clusterer = GaussianMixture(n_components=n_clusters).fit(X)
    cluster_labels = clusterer.predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouetteScore[mI] = silhouette_avg
    print("The average silhouette_score is :", silhouette_avg)
    mI = mI + 1


plt.figure()
plt.plot(list(range_n_clusters), list(silhouetteScore.values()), color='#aec7e8')
plt.ylabel("Average of Silhouette Score")
plt.xlabel("Number of clusters")
plt.title("Silhouette Score of EM")
plt.grid(True)
plt.show()

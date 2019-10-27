from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import distance
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from dunnIndex import dunn
from Dataset import loadCSV

X = loadCSV()

def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
    return d

# distance.jensenshannon()
n_clusters = 3
sse = {}
dunnIndex = {}
silhouetteScore = {}
mI = 0
# for n_clusters in range_n_clusters:
print("Estimated number of cluster: %d" % n_clusters)
# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
start_time = time.time()
clusterer = GaussianMixture(n_components=n_clusters, covariance_type='tied').fit(X)
end_time = time.time()
print("Estimated run time is %s " % (end_time - start_time))
cluster_labels = clusterer.predict(X)

# means = clusterer.means_
# print("The value of Mean %s " % means)
# covariances = clusterer.covariances_
# print("The value of covariances %s " % covariances)

# We store the K-means results in a dataframe
# pred = pd.DataFrame(cluster_labels)
# pred.columns = ['Species']
#
# # we merge this dataframe with df
# prediction = pd.concat([X, pred], axis=1)
# k_list = []
# for i in range(n_clusters):
#     clus = prediction.loc[prediction.Species == i]
#     k_list.append(clus.values)
# dunnIndex[n_clusters] = dunn(k_list)
# print("The dunn index score is %s" % dunnIndex[n_clusters])
#
# # The silhouette_score gives the average value for all the samples.
# # This gives a perspective into the density and separation of the formed
# # clusters
# silhouette_avg = silhouette_score(X, cluster_labels)
# silhouetteScore[mI] = silhouette_avg
# print("The average silhouette_score is :", silhouette_avg)
# mI = mI + 1



print(clusterer.weights_)
print(clusterer.means_)
print(clusterer.covariances_)
attr1Means = {}
attr1Cov = {}
attrs = ["xF","yF","wF","hF","xRE","yRE","xLE","yLE","xN","yN","xRM","yRM","xLM","yLM"]
for i in range(14):
    attr1Means = clusterer.means_[:, i:(i+1)].flatten()
    attr1Cov = clusterer.covariances_[:, i:(i+1)]
    print("means %s" % attr1Means)
    print("covariance %s" % attr1Cov.flatten())
    pi, mu, sigma = clusterer.weights_.flatten(), attr1Means, np.sqrt(abs(attr1Cov).flatten())
    grid = np.arange(np.min(X.iloc[:, i]), np.max(X.iloc[:, i]), 0.01)
    plt.figure()
    plt.hist(X.iloc[:, i], bins=20, density=True, alpha=0.2)
    plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label='varying weights')
    plt.xlabel("Attribute %s" % attrs[i])
    # plt.plot(grid, mix_pdf(grid, mu, sigma, [1. / n_clusters] * n_clusters), label='equal weights')
    # plt.plot(X.iloc[:, 0], [0.01] * X.iloc[:, 0].shape[0], '|', color='k')
    # plt.legend(loc='upper right')


# plt.figure()
# plt.plot(list(range_n_clusters), list(silhouetteScore.values()), color='#aec7e8')
# plt.ylabel("Average of Silhouette Score")
# plt.xlabel("Number of clusters")
# plt.grid(True)
plt.show()

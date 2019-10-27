from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from dunnIndex import dunn
from Dataset import loadCSV
import time

X = loadCSV()

# Create a subplot with 1 row and 2 columns
fig, axs = plt.subplots(2, 4)
fig.set_size_inches(16, 5)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
axs[0,0].set_xlim([-0.1, 1])
n_clusters = 3

print("K-Means - silhouette scores and formed cluster when the number of cluster is %d" % (n_clusters))
# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
start_time = time.time()
clusterer = KMeans(n_clusters=n_clusters, init='k-means++')
end_time = time.time()
print("Run time: %s" % (end_time - start_time))
cluster_labels = clusterer.fit_predict(X)

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
print('The value of Dunn Index: %s' % dunnIndex)


# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
axs[0,0].set_ylim([0, len(X) + (n_clusters + 1) * 10])

print('Estimated number of clusters: %d' % n_clusters)
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, cluster_labels)
print("The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)

y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    axs[0,0].fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    axs[0,0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

    axs[0,0].set_title("The silhouette plot for the various clusters.")
    axs[0,0].set_xlabel("The silhouette coefficient values")
    axs[0,0].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    axs[0,0].axvline(x=silhouette_avg, color="red", linestyle="--")

    axs[0,0].set_yticks([])  # Clear the yaxis labels / ticks
    axs[0,0].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed - xF and yF
    colors_1 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[0,1].scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_1, edgecolor='k')

    # axs[0,1].set_title("The visualization of the clustered data.")
    axs[0,1].set_xlabel("xF feature")
    axs[0,1].set_ylabel("yF feature")

    # 3nd Plot showing the actual clusters formed - wF and hF
    colors_2 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[0,2].scatter(X.iloc[:, 2], X.iloc[:, 3], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_2, edgecolor='k')

    # axs[0,2].set_title("The visualization of the clustered data.")
    axs[0,2].set_xlabel("wF feature")
    axs[0,2].set_ylabel("yF feature")

    # 4th Plot showing the actual clusters formed - xRE and yRE
    colors_3 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[0,3].scatter(X.iloc[:, 4], X.iloc[:, 5], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_3, edgecolor='k')

    # axs[0,3].set_title("The visualization of the clustered data.")
    axs[0,3].set_xlabel("xRE feature")
    axs[0,3].set_ylabel("yRE feature")

    # 5th Plot showing the actual clusters formed - xLE and yLE
    colors_4 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[1,0].scatter(X.iloc[:, 6], X.iloc[:, 7], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_4, edgecolor='k')

    # axs[1,0].set_title("The visualization of the clustered data.")
    axs[1,0].set_xlabel("xLE feature")
    axs[1,0].set_ylabel("yLE feature")

    # 6th Plot showing the actual clusters formed - xN and yN
    colors_5 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[1,1].scatter(X.iloc[:, 8], X.iloc[:, 9], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_5, edgecolor='k')

    # axs[1,1].set_title("The visualization of the clustered data.")
    axs[1,1].set_xlabel("xN feature")
    axs[1,1].set_ylabel("yN feature")

    # 7th Plot showing the actual clusters formed - xRM and yRM
    colors_6 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[1,2].scatter(X.iloc[:, 10], X.iloc[:, 11], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_6, edgecolor='k')

    # axs[1,2].set_title("The visualization of the clustered data.")
    axs[1,2].set_xlabel("xRM feature")
    axs[1,2].set_ylabel("yRM feature")

    # 8th Plot showing the actual clusters formed - xRM and yRM
    colors_7 = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    axs[1,3].scatter(X.iloc[:, 12], X.iloc[:, 13], marker='.', s=30, lw=0, alpha=0.7,
                c=colors_7, edgecolor='k')

    # axs[1,3].set_title("The visualization of the clustered data.")
    axs[1,3].set_xlabel("xRM feature")
    axs[1,3].set_ylabel("yRM feature")

    plt.tight_layout()

    plt.suptitle(("Silhouette analysis by K-Means with the number of cluster is %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

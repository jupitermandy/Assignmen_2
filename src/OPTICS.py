from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from Dataset import loadCSV

X = loadCSV()

clust = OPTICS(min_samples=11, metric='euclidean', cluster_method='dbscan', min_cluster_size=.05).fit(X)



space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
# ax2 = plt.subplot(G[1, 0])
# ax3 = plt.subplot(G[1, 1])
# ax4 = plt.subplot(G[1, 2])

# Reachability plot
print(labels)
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
# ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
# ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# # OPTICS
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = X[clust.labels_ == klass]
#     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
# ax2.set_title('Automatic Clustering\nOPTICS')
#
# # DBSCAN at 0.5
# colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
# for klass, color in zip(range(0, 6), colors):
#     Xk = X[labels_050 == klass]
#     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
# ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
# ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')
#
# # DBSCAN at 2.
# colors = ['g.', 'm.', 'y.', 'c.']
# for klass, color in zip(range(0, 4), colors):
#     Xk = X[labels_200 == klass]
#     ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
# ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from Dataset import loadCSV
from sklearn.neighbors import NearestNeighbors

X = loadCSV()
min_samples_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
line = [20, 21, 22, 23, 24, 25, ]
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 100, 5)
minor_ticks = np.arange(0, 100, 1)


ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2,)
ax.grid(which='major', alpha=0.5)
for min_samp in min_samples_list:
    nbrs = NearestNeighbors(n_neighbors=min_samp, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distanceDec = sorted(distances[:,min_samp-1], reverse=False)
    # plt.axhline(y=36, color='r', linestyle='-.')
    ax.plot(indices[:,0], distanceDec, color=color_sequence[min_samp-2])
plt.legend(('minPts=2', 'minPts=3',
            'minPts=4', 'minPts=5', 'minPts=6',
            'minPts=7', 'minPts=8', 'minPts=9',
            'minPts=10', 'minPts=11'),
            loc='upper left')

# plt.grid(True)
plt.ylabel("Value of eps")
plt.xlabel("Samples")
plt.title("The k-nearest neighbor distance when metric=euclidean")
plt.show()
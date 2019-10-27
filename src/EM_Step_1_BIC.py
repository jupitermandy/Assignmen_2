from scipy import linalg
from sklearn.mixture import GaussianMixture
import numpy as np
import itertools
from Dataset import loadCSV
import matplotlib.pyplot as plt
import matplotlib as mpl

X = loadCSV()

n_components_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

lowest_bic = np.infty
bic = []

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    bic.append(gmm.bic(X))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm_bic = gmm


bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
bic_bars = []

# Plot the BIC scores
plt.figure()

xpos = np.array(n_components_range)
bic_bars.append(plt.bar(xpos, bic,
                    width=.3, color='turquoise'))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.ylabel('BIC score')
plt.xlabel('Number of components')


plt.title('EM - BIC Scores')



plt.show()
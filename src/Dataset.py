import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)

def loadCSV():
    X = pd.read_csv('drivPointsedited.csv', usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], skiprows=1)
    X.head()
    desc = X.describe()
    # print(desc)
    return X

# Dataset visualization
# data = loadCSV()
# g = sns.clustermap(data, cmap="mako", robust=True)
# hm = sns.heatmap(data)
# plt.show()
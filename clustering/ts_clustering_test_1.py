import pandas as pd
from sklearn.cluster import OPTICS, KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# sns.set_style('darkgrid')

colors = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'purple',
    4: 'orange'
}

def TSAnalysis(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24)
    result.plot()
    plt.show()

def KMeansClustering(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(X)
    centers = model.cluster_centers_
    return clusters, centers

    

def OpticsClustering(X, samples=5):
    model = OPTICS(min_samples=25) #adjust minimum samples
    clusters = model.fit_predict(X)
    return clusters
    
# python3.7 ts_clustering.py water_avg_weekly.csv 100 5
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # fn = "data/Water weekly/water_avg_weekly.csv"
        fn = "data/Water weekly/water_avg_weekly_trends.csv"
        samples = 100
        n_clusters = 5
    else:
        fn = sys.argv[1]
        samples = int(sys.argv[2])
        n_clusters = int(sys.argv[3])

    df = pd.read_csv(fn, sep=',', header=None)
    X = df.iloc[:, [1, 2]]
    # X_Kmeans = df.iloc[:, 3:]

    #  First cluster by geo-coordinates
    clusters = OpticsClustering(X, samples=samples)
    values = np.unique(clusters)
    print("No. geo-coordinates clusters", len(values))

    print(clusters)

    X_ts = {}

    rcParams['axes.titlepad'] = 40

    # get geo-coordinats clusters (group of consumers within identified zones)
    for elem in values:
        X_ts[elem] = df.iloc[np.where(clusters == elem)].iloc[:, 3:]

    # cluster by each geo-coodinates cluster
    for i, elem in enumerate(X_ts):
        clusters, centers = KMeansClustering(X_ts[elem], n_clusters=n_clusters)

        # plot centers
        plt.figure(figsize=(8,6))
        for idx in range(0, n_clusters):
            plt.plot(centers[idx], label='C'+str(idx+1), color = colors[idx])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=n_clusters, mode="expand", borderaxespad=0.)
        # plt.legend(loc='right', ncol=n_clusters)
        plt.title("zone " + str(i+1) + " clusters")
  
        plt.show()

        # Analyze the TS for each center
        for idx in range(0, n_clusters):
            TSAnalysis(centers[idx])
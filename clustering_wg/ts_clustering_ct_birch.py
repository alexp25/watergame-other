import pandas as pd
from sklearn.cluster import OPTICS, KMeans, Birch
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial.distance import cdist
from statsmodels.tsa.seasonal import seasonal_decompose
import math
from sklearn.metrics import silhouette_score

from modules import graph, loader

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
    4: 'orange',
    5: 'yellow',
    6: 'gray',
    7: 'brown',
    8: 'cyan',
    9: 'olive',
    10: 'pink',
}


def getD(x1, y1, x2, y2, x3, y3):
    return abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1-x1*y2)/math.sqrt((y2-y1)**2 + (x2-x1)**2)

# Get optimal clusters for KMeans


def getOptimalClustersKmeans(data, max_clusters):
    n_clusters = range(1, max_clusters + 1)
    kmeanModels = [KMeans(n_clusters=k).fit(data).fit(data)
                   for k in n_clusters]
    distortions = [sum(np.min(cdist(data, kmeanModels[k].cluster_centers_,
                                    'euclidean'), axis=1)) / data.shape[0] for k in range(len(kmeanModels))]
    dist = {k: getD(n_clusters[0], distortions[0], n_clusters[max_clusters-1],
                    distortions[max_clusters-1], k, distortions[k-1]) for k in n_clusters}
    WCSS_vect = [kmeanModels[k].inertia_ for k in range(len(kmeanModels))]
    print(WCSS_vect)
    # plt.plot(distortions)
    # plt.show()


    fig = graph.plot(distortions, list(n_clusters), "Optimal number of clusters", "Number of clusters", "Distortion")
    graph.save_figure(fig, "./figs/eval_dist.png")
    print(distortions)
    loader.save_as_csv_1d("clusters_dist.csv", distortions)

    fig = graph.plot(WCSS_vect, list(n_clusters), "Optimal number of clusters", "Number of clusters", "WCSS")
    graph.save_figure(fig, "./figs/eval_inertia.png")
    print(WCSS_vect)
    loader.save_as_csv_1d("clusters_inertia.csv", WCSS_vect)
    quit()

    optimalClusters = max(dist, key=dist.get)
    return optimalClusters

# Get optimal clusters for Birch


def getOptimalClustersBirch(data, max_clusters):
    n_clusters = range(1, max_clusters + 1)
    birchModels = [Birch(n_clusters=k).fit(data).fit(data) for k in n_clusters]
    distortions = [sum(np.min(cdist(data, birchModels[k].cluster_centers_,
                                    'euclidean'), axis=1)) / data.shape[0] for k in range(len(birchModels))]
    dist = {k: getD(n_clusters[0], distortions[0], n_clusters[max_clusters-1],
                    distortions[max_clusters-1], k, distortions[k-1]) for k in n_clusters}
    optimalClusters = max(dist, key=dist.get)
    return optimalClusters


def TSAnalysis(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24)
    result.plot()
    plt.show()

# K-Means clustering method


def KMeansClustering(X, n_clusters=20):
    optimalClusters = getOptimalClustersKmeans(X, n_clusters)
    print("K-Means optimal clusters", optimalClusters)
    model = KMeans(n_clusters=optimalClusters)
    clusters = model.fit_predict(X)
    centers = model.cluster_centers_
    return clusters, centers, optimalClusters

# Optics clustering method


def OpticsClustering(X, samples=5):
    model = OPTICS(min_samples=25)  # adjust minimum samples
    clusters = model.fit_predict(X)
    return clusters

# Birch clustering method


def BirchClustering(X, n_clusters=20):
    optimalClusters = getOptimalClustersKmeans(X, n_clusters)
    print("Birch optimal clusters", optimalClusters)
    model = Birch(n_clusters=optimalClusters)
    clusters = model.fit_predict(X)
    centers = model.subcluster_centers_
    return clusters, centers, optimalClusters


# python3.7 ts_clustering.py water_avg_weekly.csv 100 20
if __name__ == "__main__":
    if len(sys.argv) < 2:
        fn = "data/Water weekly/water_avg_weekly.csv"
        # fn = "data/Water weekly/water_avg_weekly_trends.csv"
        samples = 100
        n_clusters = 20
    else:
        fn = sys.argv[1]
        samples = int(sys.argv[2])
        n_clusters = int(sys.argv[3])

    df = pd.read_csv(fn, sep=',', header=None)
    X = df.iloc[:, [1, 2]]
    X_Kmeans = df.iloc[:, 3:]

    # K-Means
    clusters, centers, optimalClusters = KMeansClustering(
        X_Kmeans, n_clusters=n_clusters)
    values = np.unique(clusters)
    for idx in range(0, optimalClusters):
        TSAnalysis(centers[idx])

    print(silhouette_score(X_Kmeans, clusters))

    quit()

    plt.figure()
    for idx in range(0, optimalClusters):
        plt.plot(centers[idx], label='C'+str(idx+1))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=optimalClusters, mode="expand", borderaxespad=0.)
    plt.show()
    plt.close()


    # Birch
    clusters, centers, optimalClusters = BirchClustering(
        X_Kmeans, n_clusters=n_clusters)
    values = np.unique(clusters)
    for idx in range(0, optimalClusters):
        TSAnalysis(centers[idx])
    print(silhouette_score(X_Kmeans, clusters))

    plt.figure()
    for idx in range(0, optimalClusters):
        plt.plot(centers[idx], label='C'+str(idx+1))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=optimalClusters, mode="expand", borderaxespad=0.)
    plt.show()
    plt.close()

    #  First cluster by geo-coordinates
    clusters = OpticsClustering(X, samples=samples)
    values = np.unique(clusters)
    print("No. geo-coordinates clusters", len(values))

    X_ts = {}
    for elem in values:
        X_ts[elem] = df.iloc[np.where(clusters == elem)].iloc[:, 3:]

    # cluster by each geo-coodinates cluster
    for elem in X_ts:
        clusters, centers, optimalClusters = KMeansClustering(
            X_ts[elem], n_clusters=n_clusters)
        values = np.unique(clusters)

        # plot centers
        plt.figure()
        for idx in range(0, optimalClusters):
            plt.plot(centers[idx], label='C'+str(idx+1), color=colors[idx])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=optimalClusters, mode="expand", borderaxespad=0.)
        plt.show()
        plt.close()

        # Analyze the TS for each center
        for idx in range(0, optimalClusters):
            TSAnalysis(centers[idx])

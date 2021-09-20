
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import math
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams
import matplotlib.pyplot as plt
import statistics
from sklearn.cluster import OPTICS, KMeans
from modules import clustering, graph, utils


def TSAnalysis(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24)
    result.plot()
    plt.show()


def TSAnalysisGetSeasonal(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24, extrapolate_trend='freq')
    return result.seasonal


def KMeansClustering(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = model.fit_predict(X)
    centers = model.cluster_centers_
    return clusters, centers


def OpticsClustering(X, samples=5):
    model = OPTICS(min_samples=25)  # adjust minimum samples
    clusters = model.fit_predict(X)
    return clusters


def getD(x1, y1, x2, y2, x3, y3):
    return abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1-x1*y2)/math.sqrt((y2-y1)**2 + (x2-x1)**2)


def getOptimalClusters(data, max_clusters):
    n_clusters = range(1, max_clusters + 1)
    kmeanModels = [KMeans(n_clusters=k, random_state=0).fit(data).fit(data)
                   for k in n_clusters]
    distortions = [sum(np.min(cdist(data, kmeanModels[k].cluster_centers_,
                                    'euclidean'), axis=1)) / data.shape[0] for k in range(len(kmeanModels))]
    dist = {k: getD(n_clusters[0], distortions[0], n_clusters[max_clusters-1],
                    distortions[max_clusters-1], k, distortions[k-1]) for k in n_clusters}
    optimalClusters = max(dist, key=dist.get)
    return optimalClusters


def getOptimalClustersWCSS(data, max_clusters, label):
    max_silhouette_score = 0
    silhouette_score_vect = []
    WCSS_vect = []
    optimal_number_of_clusters = 2
    n_clusters = range(1, max_clusters + 1)
    kmeanModels = [KMeans(n_clusters=k, random_state=0).fit(data)
                   for k in n_clusters]

    for km in kmeanModels:
        wcss = km.inertia_
        WCSS_vect.append(wcss)
        print(wcss)

    WCSS_vect = utils.normalize_axis_01(np.array([WCSS_vect]), 1).tolist()[0]
    fig = graph.plot(WCSS_vect, list(
        n_clusters), "Optimal number of clusters", "Number of clusters", "WCSS", True)
    graph.save_figure(fig, "./figs/eval_trends_inertia_" + label + ".png")
    optimal_number_of_clusters = getOptimalClusters(data, max_clusters)

    return optimal_number_of_clusters

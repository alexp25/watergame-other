

import time
import scipy
# from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import numpy as np


CLUSTER_MODEL = {
    "id": 0,
    "priority": 0,
    "centroid": [],
    "demand": None
}
NODE_MODEL = {
    "id": 0,
    "class": None,
    "demand": None,
    "priority": None
}

def get_array_of_arrays(a):
    array = []
    for ag in a:
        for ag1 in ag:
            array.append(ag1)
    return array

def get_centroids(self, data, n_clusters=8, init=None):
    if n_clusters is not None:
        if init is not None:
            kmeans = KMeans(n_clusters=n_clusters, init=init)
        else:
            kmeans = KMeans(n_clusters=n_clusters)
    else:
        n_clusters_range = range(2, 10)
        max_silhouette_avg = [0] * len(n_clusters_range)
        # data = np.array(data)
        for (i, k) in enumerate(n_clusters_range):
            kmeans = KMeans(n_clusters=k)
            a = kmeans.fit_predict(data)
            # print(data.shape)
            # print(a)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(data, a)
            # print("For n_clusters =", k,
            #       "The average silhouette_score is :", silhouette_avg)
            max_silhouette_avg[i] = silhouette_avg

        n_clusters = n_clusters_range[max_silhouette_avg.index(max(max_silhouette_avg))]
        kmeans = KMeans(n_clusters=n_clusters)

    a = kmeans.fit(data)
    centroids = a.cluster_centers_
    return centroids, a

def run_clustering_on_node_id(data, node_id, nclusters):
    """
    Run clustering on specified node. The data from the node is an array of arrays
    (for each day there is an array of 24 values)
    The result is the consumer behaviour over the analyzed time frame
    :param node_id:
    :param nclusters:
    :return:
    """
    t_start = time.time()

    data = data[node_id]

    if nclusters is not None and nclusters > len(data):
        print("node " + str(node_id) + "nclusters > len(data): " + str(nclusters) + "," + str(len(data)))
        return [], None, data

    res = get_centroids(data, nclusters)
    centroids = res[0]
    nc = len(centroids)
    centroids_np = np.array(centroids)
    desc = "Clusters from all data (single clustering)"
    # assign each time series to a cluster
    assignments = []

    headers = []
    for i in range(len(centroids_np)):
        headers.append("cluster " + str(i))

    # the assignments of the data series to the clusters
    assignments_series = [None] * len(assignments)
    for (i, a) in enumerate(assignments):
        assignments_series[i] = {
            "series": i,
            "cluster": int(assignments[i])
        }

    t_end = time.time()
    dt = t_end - t_start
    min = int(np.min(centroids_np))
    max = int(np.max(centroids_np))

    info = {
            "description": desc, "headers": headers,
            "dt": t_end - t_start,
            "details": {
                "node": node_id,
                "new_node": node_id,
                "n_clusters": nc,
                "n_nodes": len(data),
                "dt": int(dt * 1000),
                "min": min,
                "max": max
            },
            "assignments": assignments_series}

    return centroids_np, info, data


def get_assignments(a, data):
    return a.predict(data)        

def run_dual_clustering_on_node_range(data, nclusters, nclusters_final):
    """
        Run dual clustering on specified node range.
        The data from a node is an array of arrays
    (for each day there is an array of 24 values).
    The clusters are calculated separately for each node and added to the cluster array.
    Then, there is another clustering on this cluster array which returns
    the final clusters for all the network (consumer types in the network)
    :param r:
    :param nclusters:
    :param nclusters_final:
    :return:
    """

    t_start = time.time()
    centroid_vect = []
    raw_data_vect = []

    r = list(range(0, len(data)))

    print("node range: ", r)

    # run clustering for each node and save clusters into array
    for node_id in r:
        res = run_clustering_on_node_id(data, node_id, nclusters)
        centroid_vect.append(res[0])
        raw_data_vect.append(res[2])

    centroid_vect = get_array_of_arrays(centroid_vect)
    raw_data_vect = get_array_of_arrays(raw_data_vect)

    n_clusters_total = len(centroid_vect)
    centroids_np = np.array(centroid_vect)

    # run clustering again for the previous clusters
    res = get_centroids(centroids_np, nclusters_final, None)
    centroids = res[0]
    final_centroids = res[0]
    final_clusters = res[1]

    nc = len(centroids)
    centroids_np = np.array(centroids)

    # get assignments of time series to the final clusters
    assignments = get_assignments(res[1], raw_data_vect)

    n = len(centroids_np)
    headers = [None] * n
    clusters = []
    demands = []
    for i in range(n):
        headers[i] = "cluster " + str(i)
        cluster = CLUSTER_MODEL
        cluster["id"] = assignments[i]
        avg_demand = np.average(centroids_np[i])
        cluster["avg_demand"] = avg_demand
        demands.append(avg_demand)
        cluster["centroid"] = centroids_np[i]
        clusters.append(cluster)

    demands = np.array(demands)
    temp = demands.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(demands))

    for i in range(n):
        clusters[i]["priority"] = ranks[i]

    # print(self.clusters)
    # the assignments of the data series to the clusters
    assignments_series = [None] * len(assignments)
    for (i, a) in enumerate(assignments):
        assignments_series[i] = {
            "series": i,
            "cluster": int(a)
        }

    t_end = time.time()
    dt = t_end - t_start
    min = int(np.min(centroids_np))
    max = int(np.max(centroids_np))

    min_final = min
    max_final = max

    info = {
            "description": "Clusters from node range (dual clustering)", 
            "headers": headers,
            "dt": t_end - t_start,
            "details": {
                "node_range": r,
                "n_clusters": nc,
                "n_nodes": len(data),
                "dt": int(dt * 1000),
                "min": min,
                "max": max
            },
            "assignments": assignments_series
    }

    return centroids_np, info
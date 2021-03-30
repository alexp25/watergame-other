import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import LocalOutlierFactor


def plot_data(X):
    ### Let's see the points generated more visual. Plot the points using matplotlib
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def plot_data_with_clusters(X, kmeans, show_centers=False, xlabel=None, ylabel=None, show=True):
    ### Let's see the clusters more visual. Let's plot the graphic for our points and draw each point to the cluster it is assigned to
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow', label="points")

    if show_centers:
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="orange", s=(200, 200), label="centroids", marker="o")

    plt.title('K-Means Clustering')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.xticks()
    if show:
        plt.show()


# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
def remove_outliers(X):
    clf = LocalOutlierFactor(n_neighbors=2)
    # clf = LocalOutlierFactor()
    yhat = clf.fit_predict(X)
    print(yhat)
    # select all rows that are not outliers
    mask = yhat != -1
    clf.negative_outlier_factor_
    X = X[mask, :]
    return X


def euclid_dist(t1, t2):
    return np.sqrt(np.sum((t1 - t2) ** 2))


def k_mean_distance_2d(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances


def k_mean_distance_gen(data, center_point, i_centroid, cluster_labels):
    # maybe more easy to read
    # get the average euclidean distance for a given cluster
    data_from_cluster = data[cluster_labels == i_centroid]
    distances = []
    avg_dist = 0
    sum_dist = 0
    # calculate the distance of each point to the assigned cluster
    for data_point in data_from_cluster:
        dist = 0
        # calculate the distance on all dimensions (e.g. 2D, 3D, ...)
        for i, p in enumerate(data_point):
            dist += (p - center_point[i]) ** 2
        # dist = np.sqrt(dist)
        distances.append(dist)
        sum_dist += dist
    avg_dist = np.sqrt(sum_dist) / len(data_from_cluster)
    return distances, avg_dist, sum_dist


def clustering_kmeans(X, k):
    # kmeans = KMeans(n_clusters=k, init="random")
    # X_input = np.reshape(X, (-1, 1))
    X_input = X
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X_input)
    clusters = kmeans.predict(X_input)

    print("cluster labels: ")
    print(clusters)

    ### Let's print the cluster centers
    print("cluster centers")
    centroids = kmeans.cluster_centers_
    print(centroids)

    ### Let's print the labels for our points. In this context, label is the cluster on which the data point is assigned to after the clusterring process
    print("cluster labels")
    print(kmeans.labels_)

    # average_silhouette_score = silhouette_score(X_input, clusters)
    # print("For n_clusters =", k,
    #       "The average silhouette_score is :", average_silhouette_score)

    # average_euclid_dist = 0
    # sum_euclid_dist = 0
    # # get the distance between points that are assigned to each cluster
    # for i, point in enumerate(centroids):
    #     # get the distance for each point
    #     distance, average_euclid_dist_1, sum_euclid_dist_1 = k_mean_distance_gen(X_input, point, i, clusters)
    #     # get the average distance
    #     average_euclid_dist += average_euclid_dist_1
    #     sum_euclid_dist += sum_euclid_dist_1

    # average_euclid_dist /= len(centroids)
    # print("Inertia: ", kmeans.inertia_)
    # print("WCSS: ", sum_euclid_dist)
    # print("The average euclidean distance is :", average_euclid_dist)

    average_silhouette_score = 0
    sum_euclid_dist = 0

    return X, kmeans, average_silhouette_score, sum_euclid_dist

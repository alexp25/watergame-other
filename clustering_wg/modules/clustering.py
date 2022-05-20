import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.cm as cm
from modules import graph
from matplotlib.ticker import MaxNLocator


FSIZE_TITLE = 16
FSIZE_LABEL = 14
FSIZE_LABEL_S = 14
FSIZE_LABEL_XS = 12
OPACITY = 0.9
OPACITY = 1


def set_disp_ax(ax, title, xlabel, ylabel):
    if title:
        ax.set_title(title,  fontsize=FSIZE_TITLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FSIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FSIZE_LABEL)


def set_disp(title, xlabel, ylabel):
    if title:
        plt.gca().set_title(title, fontsize=FSIZE_TITLE, pad=10)
    if xlabel:
        plt.xlabel(xlabel, fontsize=FSIZE_LABEL)
    if ylabel:
        plt.ylabel(ylabel, fontsize=FSIZE_LABEL)


def plot_data(X):
    # Let's see the points generated more visual. Plot the points using matplotlib
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def plot_data_with_clusters(X, kmeans, show_centers=False, xlabel=None, ylabel=None, show=True, figsize=None, id=0, title="K-Means Clustering"):
    if not figsize:
        fig = plt.figure(id, figsize=(16, 8))
        # fig = plt.figure(id)
    else:
        fig = plt.figure(id, figsize=figsize)
    # Let's see the clusters more visual. Let's plot the graphic for our points and draw each point to the cluster it is assigned to
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_,
                cmap='rainbow', label="points")

    if show_centers:
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                    :, 1], color="orange", s=(200, 200), label="centroids", marker="o")

    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(loc='upper left')

    ax = plt.gca()
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    set_disp(title, xlabel, ylabel)
    plt.xticks()
    if show:
        plt.show()
    return fig


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
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                 for (x, y) in data[cluster_labels == i_centroid]]
    return distances


def k_mean_distance_gen(data, center_point, i_centroid, cluster_labels):
    # maybe more easy to read
    # get the average euclidean distance for a given cluster
    data_from_cluster = data[cluster_labels == i_centroid]
    distances = []
    avg_dist = 0
    sum_dist = 0
    use_dict = False
    # for data_point in data_from_cluster:
    #     try:
    #         enumerate(data_point)
    #     except:
    #         use_dict = True
    #         break

    # calculate the distance of each point to the assigned cluster
    for data_point in data_from_cluster:
        dist = 0

        if use_dict:
            dp = data_from_cluster[data_point]
        else:
            dp = data_point

        # calculate the distance on all dimensions (e.g. 2D, 3D, ...)
        for i, p in enumerate(dp):
            dist += (p - center_point[i]) ** 2
        # dist = np.sqrt(dist)
        distances.append(dist)
        sum_dist += dist
    avg_dist = np.sqrt(sum_dist) / len(data_from_cluster)
    return distances, avg_dist, sum_dist


def k_mean_dev_mean_distance_gen(data, i_centroid, cluster_labels):
    # maybe more easy to read
    # get the average euclidean distance for a given cluster
    data_from_cluster = data[cluster_labels == i_centroid]
    mean_data = np.mean(data, axis=0)
    # plt.plot(mean_data)
    # plt.show()
    distances = []
    avg_dist = 0
    sum_dist = 0
    # calculate the distance of each point to the assigned cluster
    for data_point in data_from_cluster:
        dist = 0
        # calculate the distance on all dimensions (e.g. 2D, 3D, ...)
        for i, p in enumerate(data_point):
            dist += (p - mean_data[i]) ** 2
        # dist = np.sqrt(dist)
        distances.append(dist)
        sum_dist += dist
    # avg_dist = np.sqrt(sum_dist) / len(data_from_cluster)
    avg_dist = np.sqrt(sum_dist) / len(data_from_cluster)
    return distances, avg_dist, sum_dist


def plot_silhouette_score(clusterer, X, cluster_labels, n_clusters):

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, (ax1) = plt.subplots(1, 1)
    # fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

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

        # color = cm.nipy_spectral(float(i) / n_clusters)
        color = cm.rainbow(float(i) / n_clusters)

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=1)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors, edgecolor='k')

    # # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')

    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')

    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    # plt.suptitle(("Silhouette analysis "
    #               "with n_clusters = %d" % n_clusters),
    #              fontsize=14, fontweight='bold')

    # plt.suptitle(("Silhouette analysis"),
    #              fontsize=14, fontweight='bold')
    graph.set_disp("Silhouette analysis",
                   "The silhouette coefficient values", "Cluster label")
    plt.show()
    graph.save_figure(fig, "./figs/silhouette.png")


def clustering_birch(X, k, same_order):
    # kmeans = KMeans(n_clusters=k, init="random")
    # X_input = np.reshape(X, (-1, 1))
    X_input = X

    model = Birch(n_clusters=k)
    clusters = model.fit_predict(X_input)
    centroids = model.subcluster_centers_

    print("cluster labels: ")
    print(clusters)

    # Let's print the cluster centers
    print("cluster centers")
    print(centroids)

    average_silhouette_score = silhouette_score(X_input, clusters)
    # print("For n_clusters =", k,
    #       "The average silhouette_score is :", average_silhouette_score)

    average_euclid_dist = 0
    sum_euclid_dist = 0
    average_euclid_dist_mean = 0
    sum_euclid_dist_mean = 0

    # get the distance between points that are assigned to each cluster
    for i, point in enumerate(centroids):
        # get the distance for each point
        distance, average_euclid_dist_1, sum_euclid_dist_1 = k_mean_distance_gen(
            X_input, point, i, clusters)
        dist_mean, average_euclid_dist_mean_1, sum_euclid_dist_mean_1 = k_mean_dev_mean_distance_gen(
            X_input, i, clusters)
        # get the average distance
        average_euclid_dist += average_euclid_dist_1
        sum_euclid_dist += sum_euclid_dist_1
        average_euclid_dist_mean += average_euclid_dist_mean_1
        sum_euclid_dist_mean += sum_euclid_dist_mean_1

    wcss = sum_euclid_dist

    # average_euclid_dist /= len(centroids)
    # print("Inertia: ", kmeans.inertia_)
    # print("WCSS: ", sum_euclid_dist)
    # print("The average euclidean distance is :", average_euclid_dist)

    # average_silhouette_score = 0
    # sum_euclid_dist = 0

    return X, model, centroids, average_silhouette_score, wcss, average_euclid_dist_mean


def clustering_kmeans_get_labels(X, k, same_order):
    # kmeans = KMeans(n_clusters=k, init="random")
    # X_input = np.reshape(X, (-1, 1))
    X_input = X
    if same_order:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0)
    else:
        kmeans = KMeans(n_clusters=k, init="k-means++")

    kmeans.fit(X_input)
    clusters = kmeans.predict(X_input)

    print("cluster labels: ")
    print(clusters)
    print("len: ", len(clusters))

    # Let's print the cluster centers
    print("cluster centers")
    centroids = kmeans.cluster_centers_
    print(centroids)
    print("len: ", len(centroids))

    # Let's print the labels for our points. In this context, label is the cluster on which the data point is assigned to after the clusterring process
    print("cluster labels")
    print(kmeans.labels_)

    return kmeans.labels_


def clustering_kmeans(X, k, same_order):
    # kmeans = KMeans(n_clusters=k, init="random")
    # X_input = np.reshape(X, (-1, 1))
    X_input = X
    if same_order:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0)
    else:
        kmeans = KMeans(n_clusters=k, init="k-means++")

    kmeans.fit(X_input)
    clusters = kmeans.predict(X_input)

    print("cluster labels: ")
    print(clusters)
    print("len: ", len(clusters))

    # Let's print the cluster centers
    print("cluster centers")
    centroids = kmeans.cluster_centers_
    print(centroids)
    print("len: ", len(centroids))

    # Let's print the labels for our points. In this context, label is the cluster on which the data point is assigned to after the clusterring process
    print("cluster labels")
    print(kmeans.labels_)

    # plot_silhouette_score(kmeans, X_input, clusters, k)

    average_silhouette_score = silhouette_score(X_input, clusters)
    # print("For n_clusters =", k,
    #       "The average silhouette_score is :", average_silhouette_score)

    average_euclid_dist = 0
    sum_euclid_dist = 0
    average_euclid_dist_mean = 0
    sum_euclid_dist_mean = 0

    sum_euclid_dist_each = []

    # get the distance between points that are assigned to each cluster
    for i, point in enumerate(centroids):
        # get the distance for each point
        distance, average_euclid_dist_1, sum_euclid_dist_1 = k_mean_distance_gen(
            X_input, point, i, clusters)
        dist_mean, average_euclid_dist_mean_1, sum_euclid_dist_mean_1 = k_mean_dev_mean_distance_gen(
            X_input, i, clusters)
        # get the average distance
        average_euclid_dist += average_euclid_dist_1
        sum_euclid_dist += sum_euclid_dist_1
        sum_euclid_dist_each.append(sum_euclid_dist_1)
        average_euclid_dist_mean += average_euclid_dist_mean_1
        sum_euclid_dist_mean += sum_euclid_dist_mean_1

    wcss = sum_euclid_dist
    wcss = kmeans.inertia_

    # average_euclid_dist /= len(centroids)
    # print("Inertia: ", kmeans.inertia_)
    # print("WCSS: ", sum_euclid_dist)
    # print("The average euclidean distance is :", average_euclid_dist)

    # average_silhouette_score = 0
    # sum_euclid_dist = 0

    return X, kmeans, centroids, average_silhouette_score, wcss, average_euclid_dist_mean, sum_euclid_dist_each

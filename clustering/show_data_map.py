import pandas as pd
from sklearn.cluster import OPTICS, KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams
import matplotlib
from modules import graph
import statistics


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

labels = ["A", "B", "C", "D"]
sizes = [10, 50, 100, 200]
sizes.reverse()


def TSAnalysis(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24)
    result.plot()
    plt.show()


def TSAnalysisGetSeasonal(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24, extrapolate_trend='freq')
    return result.seasonal


def KMeansClustering(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(X)
    centers = model.cluster_centers_
    return clusters, centers


def OpticsClustering(X, samples=5):
    model = OPTICS(min_samples=25)  # adjust minimum samples
    clusters = model.fit_predict(X)
    return clusters


# python3.7 ts_clustering.py water_avg_weekly.csv 100 5
if __name__ == "__main__":
    if len(sys.argv) < 2:
        fn = "data/Water weekly/water_avg_weekly.csv"
        # fn = "data/Water weekly/water_avg_weekly_trends.csv"
        samples = 100
        n_clusters = 4
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

    # print(clusters)
    plot_centers = False
    plot_seasonal = False
    plot_hist = True

    X_ts = {}
    X_coords = {}
    X_centers = {}
    X_centers_vect = []
    scale_coords = 100000

    X_coords_clusters = {}

    stdev_clusters = {}
    stdev_coords = {}

    X_containers = {}

    # get geo-coordinats clusters (group of consumers within identified zones)
    for elem in values:
        X_ts[elem] = df.iloc[np.where(clusters == elem)].iloc[:, 3:]
        X_coords[elem] = df.iloc[np.where(clusters == elem)].iloc[:, :3]

        X_centers[elem] = (np.average(X_coords[elem][1])/scale_coords,
                           np.average(X_coords[elem][2])/scale_coords)

        X_containers[elem] = df.iloc[np.where(clusters == elem)].iloc[:, 1:]

        X_centers_vect.append(X_centers[elem])
        stdev_clusters[elem] = []
        stdev_coords[elem] = []

    rcParams['axes.titlepad'] = 40

    # cluster by each geo-coodinates cluster
    for i, elem in enumerate(X_ts):
        clusters, centers = KMeansClustering(X_ts[elem], n_clusters=n_clusters)
        # print(clusters)
        X_coords_clusters[elem] = {}
        X_containers[elem].insert(0, "clusters", clusters)
        X_containers[elem].insert(0, "stdev", clusters)

        norm = np.linalg.norm(centers)
        centers = centers / norm

        # plot centers
        if plot_centers:
            plt.figure(figsize=(8, 6))
            for idx in range(0, n_clusters):
                plt.plot(centers[idx], label='C'+str(idx+1), color=colors[idx])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                       ncol=n_clusters, mode="expand", borderaxespad=0.)
            # plt.legend(loc='right', ncol=n_clusters)
            plt.title("zone " + str(i+1) + " clusters")
            plt.show()

        stdev_clusters_1 = []
        stdev_coords_1 = []

        # Analyze the TS for each center
        for idx in range(0, n_clusters):
            result = TSAnalysisGetSeasonal(centers[idx])
            stdev = statistics.stdev(result)
            # print(stdev)
            stdev_clusters_1.append(stdev)

            # get average coords for cluster
            df = X_containers[elem]
            coords = df.iloc[np.where(df["clusters"] == idx)]
            # add standard deviation to cluster elements
            df.loc[coords.index, "stdev"] = stdev

            avg_coords = (np.average(
                coords.iloc[:, 2])/scale_coords, np.average(coords.iloc[:, 3])/scale_coords)

            # cluster coords from original points/assignments
            stdev_coords_1.append(avg_coords)

            if plot_seasonal:
                plt.plot(result)
                plt.show()

        stdev_clusters[elem] = stdev_clusters_1
        stdev_coords[elem] = stdev_coords_1

    print(stdev_clusters)
    print(stdev_coords)

    stdev_coords_vect = []
    for d in stdev_coords:
        stdev_coords_vect = stdev_coords_vect + stdev_coords[d]

    print(X_containers)

    stdev_min = []
    stdev_max = []

    # get min max stdev
    for sd in stdev_clusters:
        sdc = stdev_clusters[sd]
        stdev_min.append(np.min(sdc))
        stdev_max.append(np.max(sdc))

    stdev_min = np.min(stdev_min)
    stdev_max = np.max(stdev_max)

    stdev_min *= 1.1
    stdev_max *= 1.1

    # compute levels for stdev reclustering
    levels = np.linspace(stdev_min, stdev_max, n_clusters)
    print(stdev_min, stdev_max)
    print(levels)
    # levels = [0.001, 0.01, 0.02, 0.05]
    levels = -np.sort(-levels)

    # recluster by stdev level
    stdev_coords_by_stdev = {}
    for level_idx, level in enumerate(levels):
        stdev_coords_by_stdev[str(level_idx)] = []

    new_assignments = []
    for k in stdev_clusters:
        for i, elem in enumerate(stdev_clusters[k]):
            elem_adj = "0"
            for level_idx, level in enumerate(levels):
                if elem < level:
                    elem_adj = str(len(levels) - level_idx - 1)
            new_assignments.append(int(elem_adj))
            print(elem_adj)
            elem_coord = stdev_coords[k][i]
            stdev_coords_by_stdev[elem_adj].append(elem_coord)

    # print(stdev_coords_by_stdev)
    # print("\n\n")

    print(new_assignments)
    # labels = [str(level) for level in range(len(levels))]
    # labels

    d = {}
    for elem in new_assignments:  # pass through all the characters in the string
        if d.get(elem):  # verify if the character exists in the dictionary
            d[elem] += 1  # if it exist add 1 to the value for that character
        else:  # if it doesn’t exist initialize a new key with the value of the character
            d[elem] = 1  # and initialize the value (which is the counter) to 1

    d_vect = []
    for level in range(len(levels)):
        d_vect.append(d[level])

    # cmap = matplotlib.cm.get_cmap('viridis')
    # color_scheme = [cmap(i) for i in np.linspace(0, 1, len(levels))]
    # print(color_scheme)

    colors_map = graph.create_discrete_cmap(levels)
    # colors_map = matplotlib.cm.get_cmap('viridis')
    colors_plot = [colors_map(i+1) for i in range(len(levels))]
    colors_plot = list(reversed(colors_plot))
    # colors_plot = list(colors_plot)

    split_d_vect = []
    for i, d in enumerate(d_vect):
        d_vect_new = []
        for i1, d1 in enumerate(d_vect):
            if i1 == i:
                d_vect_new.append(d)
            else:
                d_vect_new.append(0)
        split_d_vect.append(d_vect_new)

    if plot_hist:
        # fig = graph.plot_barchart(labels, d_vect, "Cluster", "Assignments", "Clustering distribution", "b", [0, 18], None)
        fig, _ = graph.plot_barchart_multi_core_raw(split_d_vect, colors_plot, labels, "Cluster", "Assignment count",
                                                    "Clustering distribution", None, [0, 18], True, None, 0, None)

        graph.save_figure(fig, "./figs/map_cluster_hist.png")

    series = [str(i+1) for i in range(len(X_centers_vect))]

    print("\n")
    print(X_centers_vect)

    fig = graph.plot_map_features(X_centers_vect, stdev_coords_by_stdev,
                            colors_plot, None, None, 0.1, labels, sizes)

    graph.save_figure(fig, "./figs/map_cluster_result.png")
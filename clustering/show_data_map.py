import pandas as pd
from sklearn.cluster import OPTICS, KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams
import matplotlib
from modules import graph, utils
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
labels.reverse()
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
    model = KMeans(n_clusters=n_clusters, random_state=0)
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
    plot_average_stdev = False
    split_stdev_by_zone = True

    X_ts = {}
    X_coords = {}
    X_centers = {}
    X_centers_vect = []
    scale_coords = 100000

    X_coords_clusters = {}

    stdev_clusters = {}
    stdev_clusters_avg = {}
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

        # plot centers
        if plot_centers:
            # plt.figure(figsize=(8, 6))
            fig = plt.figure()
            graph.set_plot_font()
            for idx in range(0, n_clusters):
                plt.plot(centers[idx], label='C'+str(idx+1), color=colors[idx])
            # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            #            ncol=n_clusters, mode="expand", borderaxespad=0.)
            plt.legend(loc="upper left", fontsize=graph.FSIZE_LABEL_XS)
            # plt.legend(loc='right', ncol=n_clusters)
            plt.grid(zorder=0)
            graph.set_disp("zone " + str(i+1) + " clusters",
                           "x [hours]", "y [L/h]")
            plt.show()
            graph.save_figure(
                fig, "./figs/map_cluster_zone_" + str(i+1) + ".png")

        norm = np.linalg.norm(centers)
        centers = centers / norm

        stdev_clusters_1 = []
        stdev_coords_1 = []

        # Analyze the TS for each center
        for idx in range(0, n_clusters):
            # TSAnalysis(centers[idx])
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
                fig = graph.plot(result, None, "zone " + str(i+1) + " cluster " + str(idx) + " seasonal",
                                 "x [hours]", "y [normalized]", False)
                graph.save_figure(
                    fig, "./figs/map_cluster_seasonal_"+str(i+1)+"_" + str(idx) + ".png")

        stdev_clusters_avg[elem] = np.mean(stdev_clusters_1)
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

    levels_zones = {}

    # get min max stdev
    for sd in stdev_clusters:
        sdc = stdev_clusters[sd]
        stdev_min_zone = np.min(sdc)
        stdev_max_zone = np.max(sdc)
        levels_zones[sd] = utils.get_levels(stdev_min_zone, stdev_max_zone, n_clusters + 1)
        stdev_min.append(stdev_min_zone)
        stdev_max.append(stdev_max_zone)

    stdev_min = np.min(stdev_min)
    stdev_max = np.max(stdev_max)

    # compute levels for stdev reclustering
    levels = utils.get_levels(stdev_min, stdev_max, n_clusters + 1)
    print(levels)

    # recluster by stdev level global
    stdev_coords_by_stdev, new_assignments = utils.assign_levels(
        stdev_clusters, stdev_coords, levels)
    d_vect = utils.check_hist(new_assignments, levels)

    # recluster by stdev level by zone
    stdev_coords_by_stdev_by_zone, new_assignments_by_zone = utils.assign_levels_by_zones(
        stdev_clusters, stdev_coords, levels_zones)
    d_vect_zone = utils.check_hist(new_assignments_by_zone, levels)
    print(levels_zones)
    print(new_assignments_by_zone)
    print(stdev_coords_by_stdev_by_zone)

    print(d_vect)
    print(d_vect_zone)
    # quit()

    colors_map = graph.create_discrete_cmap(levels)
    colors_plot = [colors_map(i+1) for i in range(len(levels))]
    colors_plot = list(reversed(colors_plot))

    max_d = 0
    if split_stdev_by_zone:
        split_d_vect = utils.get_split_d_vect(d_vect_zone)
        max_d = np.max(d_vect_zone) + 1
    else:
        split_d_vect = utils.get_split_d_vect(d_vect)
        max_d = np.max(d_vect) + 1
  
    if plot_hist:
        fig, _ = graph.plot_barchart_multi_core_raw(split_d_vect, colors_plot, labels, "Cluster", "Assignment count",
                                                    "Clustering distribution", None, [0, max_d], True, None, 0, None)
        if split_stdev_by_zone:
            graph.save_figure(fig, "./figs/map_cluster_hist_zones.png")
        else:
            graph.save_figure(fig, "./figs/map_cluster_hist.png")

    series = [str(i+1) for i in range(len(X_centers_vect))]

    if split_stdev_by_zone:
        fig = graph.plot_map_features(X_centers_vect, stdev_coords_by_stdev_by_zone,
                                    colors_plot, None, None, 0.1, labels, sizes, "Map results")       
        graph.save_figure(fig, "./figs/map_cluster_result_zones.png")
    else:
        if plot_average_stdev:
            stdev_coords_by_stdev = utils.assign_levels_1d(
                stdev_clusters_avg, X_centers_vect, levels)
        fig = graph.plot_map_features(X_centers_vect, stdev_coords_by_stdev,
                                  colors_plot, None, None, 0.1, labels, sizes, "Map results")
        if plot_average_stdev:
            graph.save_figure(fig, "./figs/map_cluster_result_global.png")
        else:
            graph.save_figure(fig, "./figs/map_cluster_result.png")

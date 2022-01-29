

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np


def run_clustering(x, nc, xheader, xlabels=None):
    if nc is None:
        # use silhouette score
        max_silhouette_score = 0
        silhouette_score_vect = []
        WCSS_vect = []
        optimal_number_of_clusters = 2
        r = range(2, 20)
        for nc1 in r:
            X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean = clustering.clustering_kmeans(
                x, nc1, True)
            silhouette_score_vect.append(silhouette_score)
            WCSS_vect.append(WCSS)
            # WCSS_vect.append(average_euclid_dist_mean)
            if silhouette_score > max_silhouette_score:
                max_silhouette_score = silhouette_score
                optimal_number_of_clusters = nc1
        nc = optimal_number_of_clusters
        fig = graph.plot(silhouette_score_vect, list(
            r), "Optimal number of clusters", "Number of clusters", "Silhouette score", True)
        WCSS_vect = utils.normalize_axis_01(
            np.array([WCSS_vect]), 1).tolist()[0]
        fig = graph.plot(WCSS_vect, list(
            r), "Optimal number of clusters", "Number of clusters", "WCSS", True)
        graph.save_figure(fig, "./figs/eval_trends_inertia.png")
        X, kmeans, centroids, silhouette_score, _, _ = clustering.clustering_kmeans(
            x, nc, True)
        print("optimal number of clusters: " + str(nc) +
              " (" + str(max_silhouette_score) + ")")
    else:
        X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean = clustering.clustering_kmeans(
            x, nc, True)
        # X, kmeans, centroids, avg_dist, sum_dist, average_euclid_dist_mean = clustering.clustering_birch(x, nc, True)

    # print(avg_dist)
    # print(sum_dist)
    # print(average_euclid_dist_mean)

    print("silhouette score: ", silhouette_score)

    # quit()

    # quit()
    xc = np.transpose(centroids)
    # xc = np.transpose(xc)

    if xlabels is not None:
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

    sx = np.shape(xc)
    print(sx)
    tss = utils.create_timeseries(xc, xheader, xlabels)
    return tss, nc


nc = 4

root_data_folder = "./data"
# read the data from the csv file

rolling_filter = False
# rolling_filter = True

result_name = root_data_folder + "/res"
if rolling_filter:
    result_name += "_rf"
result_name += ".csv"

plot_all_data = True
# plot_all_data = False

rolling_filter = True
rolling_filter = False

start_index = 1
# end_index = 100
end_index = None
start_col = 3
end_col = None
fill_start = False

filter_labels = []
# filter_labels = ["toaleta", "chiuveta_rece", "chiuveta_calda"]
filter_labels = ["toaleta"]

x, header = loader.load_dataset(result_name)

df = loader.load_dataset_pd(result_name)

if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]
    # df = df[df['label'] == filter_labels]

x = df.to_numpy()
print(x)

nheader = len(header)

sx = np.shape(x)

print(sx)
print(nheader)
header = []
for d in range(sx[0]):
    # header.append(str(d+1))
    header.append(x[d, 0] + " - " + x[d, 1])
print(header)
# quit()

if fill_start:
    x = x[start_index:, :]
    x[:, 0:start_col-1] = np.transpose(np.array([[0] * (sx[0]-1)]))
else:
    x = np.transpose(x[start_index:, start_col:])

if end_index is not None:
    x = x[:end_index, :]
if end_col is not None:
    x = x[:, :end_col]


xlabel = "flow [L/h]"
xtype = "flow"
title = "consumption data (flow)"

sx = np.shape(x)
print(sx)

print("start")


# time axis labels
xlabels = [str(i) for i in range(sx[0])]
# xlabels_disp = [str(i%1440) for i in range(sx[0])]
xlabels_disp = xlabels
# xlabels = [str(i) for i in range(len(timestamps))]
# xlabels = timestamps
# xlabels = [np.datetime64(ts) for ts in timestamps]

# print(xlabels)
# quit()
# xlabels = [str(e) for e in list(range(nheader-3))]
# xlabels = None

# xlabels = np.array(xlabels)
# xlabels = np.transpose(xlabels)
# print(xlabels)

if plot_all_data:
    xplot = x
    xlabels_disp = [xlabels_disp] * sx[1]
    if xlabels_disp is not None:
        xlabels_disp = np.array(xlabels_disp)
        xlabels_disp = np.transpose(xlabels_disp)
    tss = utils.create_timeseries(xplot, header, xlabels_disp)
    fig = graph.plot_timeseries_multi_sub2(
        [tss], [title], "x [samples]", [xlabel], (16, 9), 14, None, None, True, 0)
    result_name = "./figs/consumer_data"
    graph.save_figure(fig, result_name)

xlabels_disp = xlabels
# cluster labels
xheader = ["c" + str(i+1) for i in range(sx[1])]

if xlabels_disp is not None:
    if nc is None:
        xlabels_disp = [xlabels_disp] * 1000
    else:
        xlabels_disp = [xlabels_disp] * nc

x = np.transpose(x)

tss, nc = run_clustering(x, nc, xheader, xlabels_disp)

# plot cluster centroids
title = "weekly consumer patterns (" + str(nc) + "c)"

ylabel = "y [L/h]"

fig = graph.plot_timeseries_multi_sub2(
    [tss], [title], "x [samples]", [ylabel], (16, 9), 14, None, None, True, 1)

result_name = "./figs/consumer_patterns_" + str(nc) + "c"

graph.save_figure(fig, result_name)

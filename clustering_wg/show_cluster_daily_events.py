

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np
import statistics


def run_clustering(x, nc, xheader, xlabels=None):
    if nc is None:
        # use silhouette score
        max_silhouette_score = 0
        silhouette_score_vect = []
        WCSS_vect = []
        optimal_number_of_clusters = 2
        r = range(2, 20)
        for nc1 in r:
            X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean, _ = clustering.clustering_kmeans(
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
        X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean, _ = clustering.clustering_kmeans(
            x, nc, True)
        # X, kmeans, centroids, avg_dist, sum_dist, average_euclid_dist_mean = clustering.clustering_birch(x, nc, True)

    print("silhouette score: ", silhouette_score)
    xc = np.transpose(centroids)

    if xlabels is not None:
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

    sx = np.shape(xc)
    print(sx)
    tss = utils.create_timeseries(xc, xheader, xlabels)
    return tss, nc, xc


nc = 1

root_data_folder = "./data"
# read the data from the csv file

rolling_filter = False
# rolling_filter = True

res_name = "res"

result_name = root_data_folder + "/" + res_name
if rolling_filter:
    result_name += "_rf"
result_name += ".csv"


result_ts_name = root_data_folder + "/" + res_name + "_ts"
if rolling_filter:
    result_ts_name += "_rf"
result_ts_name += ".csv"

plot_all_data = True
plot_all_data = False

rolling_filter = True
rolling_filter = False

start_index = 0
# end_index = 100
end_index = 20160  # 2 weeks in minutes
start_col = 3
end_col = None
fill_start = False
n_days = 14

filter_labels = []
# filter_labels = ["toaleta", "chiuveta_rece", "chiuveta_calda", "dus"]
# filter_labels = ["toaleta"]
filter_uid = ["41364_1", "41364_2", "41364_3", "41364_4"]
# filter_uid = []

x, header = loader.load_dataset(result_name)
df = loader.load_dataset_pd(result_name)

t, _ = loader.load_dataset(result_ts_name)
df_ts = loader.load_dataset_pd(result_ts_name)

x_ts = df_ts.to_numpy()

print(x_ts)

if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]
if len(filter_uid) > 0:
    boolean_series = df['uid'].isin(filter_uid)
    df = df[boolean_series]

x = df.to_numpy()
print(x)

# quit()

nheader = len(header)

sx = np.shape(x)

print(sx)

print(nheader)
header = []
for d in range(sx[0]):
    header.append(x[d, 0] + " - " + x[d, 1])
print(header)

if fill_start:
    x = x[start_index:, :]
    x[:, 0:start_col-1] = np.transpose(np.array([[0] * (sx[0]-1)]))
else:
    x = np.transpose(x[start_index:, start_col:])
    x_ts = np.transpose(x_ts[start_index:, start_col:])

if end_index is not None:
    x = x[:end_index, :]
    x_ts = x_ts[:end_index, :]
if end_col is not None:
    x = x[:, :end_col]
    x_ts = x_ts[:, :end_col]

for consumer in range(sx[0]):
    stdev = statistics.stdev(x[:, consumer])
    print("stdev ", consumer, " = ", stdev)
# quit()

x_split = np.split(x, n_days)
x_ts_split = np.split(x_ts, n_days)

print(x_ts)

# quit()

# print(x_split)


xlabel = "flow [L/h]"
xtype = "flow"
title = "consumption data (flow)"

sx = np.shape(x)
print(sx)

print("start")

# time axis labels
xlabels = [str(i) for i in range(sx[0])]
xlabels_disp = xlabels

# header days
header = [str(i) for i in range(len(x_split))]
header = xlabels

x_consumer_average_vect = []

# for day in range(n_days):
for consumer in range(sx[1]):
    x_consumer = []
    for days in x_split:
        print(np.shape(days))
        x_consumer.append(days[:, consumer])
    x_consumer = np.transpose(np.array(x_consumer))

    print(x_consumer)

    sx = np.shape(x_consumer)

    if plot_all_data:
        xplot = x_consumer
        xlabels_disp = [xlabels_disp] * sx[1]
        if xlabels_disp is not None:
            xlabels_disp = np.array(xlabels_disp)
            xlabels_disp = np.transpose(xlabels_disp)
        tss = utils.create_timeseries(xplot, header, None, None)
        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "x [samples]", [xlabel], (8, 6), 14, None, None, True, 0)
        result_name = "./figs/consumer_data"
        graph.save_figure(fig, result_name, 200)

    xlabels_disp = xlabels
    # cluster labels
    xheader = ["c" + str(i+1) for i in range(sx[1])]

    if xlabels_disp is not None:
        if nc is None:
            xlabels_disp = [xlabels_disp] * 1000
        else:
            xlabels_disp = [xlabels_disp] * nc

    x_consumer_input = np.transpose(x_consumer)

    if nc == 1:
        # average rows
        x_consumer_average = x_consumer_input.mean(axis=0)
        x_consumer_average_vect.append(x_consumer_average)
    else:
        tss, nc, xc = run_clustering(
            x_consumer_input, nc, xheader, xlabels_disp)
        xc = np.transpose(xc)
        print(xc)
        xc = xc.tolist()
        for c in xc:
            x_consumer_average_vect.append(c)
        # plot cluster centroids
        # title = "consumer patterns (" + str(nc) + "c)"
        # ylabel = "y [L/h]"
        # fig = graph.plot_timeseries_multi_sub2(
        #     [tss], [title], "x [samples]", [ylabel], (8, 6), 10, None, None, True, 1)

        # result_name = "./figs/consumer_patterns_day_" + str(nc) + "c"
        # graph.save_figure(fig, result_name, 200)

print(x_consumer_average_vect)
header = ["chiuveta_rece", "chiuveta_calda", "toaleta", "dus"]
x_consumer_average_vect = np.array(x_consumer_average_vect)
x_consumer_average_vect = np.transpose(x_consumer_average_vect)

# adjust for starting time, roll to start with 0
x_consumer_average_vect = np.roll(x_consumer_average_vect, 14*60, axis=0)

datax = [str(h) for h in list(range(0, 24*60))]
datax_labels = [str(h//60) for h in list(range(0, 24*60))]
datax_labels = ["0"+str(h % 24) if h < 10 else str(h % 24)
                for h in list(range(0, 24))]  # original time in UTC (data rolled)
tss = utils.create_timeseries(x_consumer_average_vect, header, None, datax)
figsize = (12, 6)
figsize = (8, 6)
fig = graph.plot_timeseries_multi_sub2(
    [tss], ["Daily consumption events"], "x [time of day]", [xlabel], figsize, 24, None, datax_labels, True, 0)
result_name = "./figs/consumer_patterns_day_" + str(nc) + "c"
graph.save_figure(fig, result_name, 200)

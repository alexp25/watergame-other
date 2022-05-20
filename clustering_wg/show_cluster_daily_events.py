

# import our modules
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np
import statistics

import yaml
config = yaml.safe_load(open("config.yml"))


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
    return tss, nc, xc, kmeans


nc = 1

root_data_folder = "./data"
# read the data from the csv file

rolling_filter = False
# rolling_filter = True

show_actual_classes = True
show_actual_classes = False

process_daily_batches = False
# process_daily_batches = True

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

start_index = 0
# end_index = 100
end_index = 20160  # 2 weeks in minutes
start_col = 3
end_col = None
fill_start = False
n_days = 14
hours_roll = 14

selection = "all"

fname_dict = config["fname_dict"]
title_dict = config["title_dict"]

fname = selection
filter_labels = fname_dict[selection]

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

nheader = len(header)

sx = np.shape(x)

print(sx)

print(nheader)
header = []
header_groups = []
for d in range(sx[0]):
    header.append(x[d, 0] + " - " + x[d, 1])
    header_groups.append(x[d, 1])
print(header)
loc_header = header
print(header_groups)


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

# x_split = preprocessing.remove_fit_max_cols(x_split)

print(x_ts)

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
x_consumer_input_combined = None
x_consumer_average_groups = {}

actual_labels = []
d = {'chiuveta_rece': 1, 'chiuveta_calda': 2, "toaleta": 3, "dus": 4, "masina_spalat": 5, "masina_spalat_vase": 6}

# for day in range(n_days):
for consumer in range(sx[1]):
    x_consumer = []
    for days in x_split:
        print(np.shape(days))
        x_consumer.append(days[:, consumer])
        actual_labels.append(d[header_groups[consumer]])
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
    print(np.shape(x_consumer_input))

    if x_consumer_input_combined is None:
        x_consumer_input_combined = x_consumer_input
    else:
        x_consumer_input_combined = np.concatenate(
            (x_consumer_input_combined, x_consumer_input), axis=0)

    x_consumer_average = x_consumer_input.mean(axis=0)
    if not header_groups[consumer] in x_consumer_average_groups:
        x_consumer_average_groups[header_groups[consumer]] = [
            x_consumer_average]
    else:
        x_consumer_average_groups[header_groups[consumer]].append(
            x_consumer_average)   
    
    if nc == 1:
        # average rows
        x_consumer_average_vect.append(x_consumer_average)
    else:
        tss, nc, xc, kmeans = run_clustering(
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

datax = [str(h) for h in list(range(0, 24*60))]
datax_labels = [str(h//60) for h in list(range(0, 24*60))]
datax_labels = ["0"+str(h % 24) if h < 10 else str(h % 24)
                for h in list(range(0, 24))]  # original time in UTC (data rolled)

if process_daily_batches:
    print("actual labels: ", actual_labels)
    # quit()
    if show_actual_classes:
        x_consumer_input_combined = []    
        for g in x_consumer_average_groups.keys():
            a = np.average(np.array(x_consumer_average_groups[g]), axis=0)
            print(a)
            x_consumer_input_combined.append(a)              
            
        xc = np.array(x_consumer_input_combined)
        xplot = np.transpose(xc)
        xplot = np.roll(xplot, hours_roll*60, axis=0)
        if xlabels is not None:
            xlabels = np.array(xlabels)
            xlabels = np.transpose(xlabels)
        print(xlabels)
        xlabels = None
        header = [config["name_mapping"][key] for key in list(x_consumer_average_groups.keys())]
        print(np.shape(xplot))
        tss = utils.create_timeseries(xplot, header, xlabels)
    else:
        x_consumer_input_combined = np.array(x_consumer_input_combined)
        x_consumer_input_combined = np.transpose(x_consumer_input_combined)
        # adjust for starting time, roll to start with 0
        x_consumer_input_combined = np.roll(
            x_consumer_input_combined, hours_roll*60, axis=0)
        x_consumer_input_combined = np.transpose(x_consumer_input_combined)
        print(x_consumer_input_combined)
        print(np.shape(x_consumer_input_combined))
        sx = np.shape(x_consumer_input_combined)
        xheader = ["c" + str(i+1) for i in range(sx[1])]
        # time axis labels
        xlabels = [str(i) for i in range(sx[0])]
        xlabels_disp = xlabels
        xlabels_disp = None
        tss, nc, xc, kmeans = run_clustering(
            x_consumer_input_combined, len(filter_labels), xheader, xlabels_disp)
        print("cluster labels: ", list(kmeans.labels_))
        xc = np.transpose(xc)
        print(xc)
        xc = xc.tolist()

    figsize = (12, 6)
    figsize = (8, 6)
    fig = graph.plot_timeseries_multi_sub2(
        [tss], ["Daily consumption patterns"], "time of day [h]", [xlabel], figsize, 24, None, datax_labels, True, 0)
    result_name = "./figs/consumer_patterns_day_all_" + str(nc) + "c"
    if rolling_filter:
        result_name += "_rf"
    if show_actual_classes:
        result_name += "_actual"
    graph.save_figure(fig, result_name, 200)
else:
    print(x_consumer_average_vect)
    # header = [config["name_mapping"][loc] for loc in filter_labels]
    header = loc_header

    x_consumer_average_vect = np.array(x_consumer_average_vect)
    x_consumer_average_vect = np.transpose(x_consumer_average_vect)

    # adjust for starting time, roll to start with 0
    x_consumer_average_vect = np.roll(
        x_consumer_average_vect, hours_roll*60, axis=0)

    datax = [str(h) for h in list(range(0, 24*60))]
    datax_labels = [str(h//60) for h in list(range(0, 24*60))]
    datax_labels = ["0"+str(h % 24) if h < 10 else str(h % 24)
                    for h in list(range(0, 24))]  # original time in UTC (data rolled)
    tss = utils.create_timeseries(x_consumer_average_vect, header, None, datax)
    figsize = (12, 6)
    figsize = (8, 6)
    fig = graph.plot_timeseries_multi_sub2(
        [tss], ["Daily consumption patterns"], "time of day [h]", [xlabel], figsize, 24, None, datax_labels, True, 0)
    result_name = "./figs/consumer_patterns_day_" + str(nc) + "c"
    graph.save_figure(fig, result_name, 200)

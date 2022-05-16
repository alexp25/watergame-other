

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np
import matplotlib.pyplot as plt
import statistics

import yaml
config = yaml.safe_load(open("config.yml"))


root_data_folder = "./data"
# read the data from the csv file

rolling_filter = False
# rolling_filter = True

result_name = root_data_folder + "/res_evt"
if rolling_filter:
    result_name += "_rf"
result_name += ".csv"

plot_all_data = True
# plot_all_data = False

rolling_filter = True
rolling_filter = False

start_index = 0
# end_index = 100
end_index = None
start_col = 3
end_col = 3
fill_start = False

savefig = True
savefile = True

nc = 3

eval_nc = True
eval_nc = False

selection = "all"

fname_dict = config["fname_dict"]
title_dict = config["title_dict"]

fname = selection
filter_labels = fname_dict[selection]
filter_uid = ["41364_1", "41364_2", "41364_3", "41364_4"]
filter_uid = []

# x, header = loader.load_dataset(result_name)
df = loader.load_dataset_pd(result_name)

if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]
    # df["duration"] > 0
    df = df[df["volume"] >= 1]
    df = df[df["duration"] > 0]
    df = df[df["duration"] < 10]  
if len(filter_uid) > 0:
    boolean_series = df['uid'].isin(filter_uid)
    df = df[boolean_series]

df.drop("timestamp", inplace=True, axis=1)

print(df)

x = df.to_numpy()
x = x[start_index:, start_col:]

sx = np.shape(x)

for filter_uid_1 in filter_uid:
    boolean_series = df['uid'].isin([filter_uid_1])
    df1 = df[boolean_series]
    stdev = statistics.stdev(df1["volume"] * df1["duration"])
    print("stdev ", filter_uid_1, " = ", stdev)

# quit()
# x = x[:,:1]

print(x)
print(np.shape(x))
# quit()

# run the k-means algorithm with nc (number of continents) clusters
X, kmeans, centroids, average_silhouette_score, wcss, average_euclid_dist_mean, sum_euclid_dist_each = clustering.clustering_kmeans(
    x, nc, True)

print(centroids)
# sort by "duration"
sort_order = centroids[:, 0].argsort()
print(sort_order)
centroids = centroids[sort_order]
print(centroids)

print(kmeans.labels_)
unique_labels = list(set(kmeans.labels_))
print(unique_labels)
# print(sort_order)
# unique_labels = [unique_labels[i] for i in sort_order]
# print(unique_labels)

# swap labels to match sort_order
labels = kmeans.labels_
dict_swap = []
for i, so in enumerate(sort_order):
    dict_swap.append(so)
labels = [dict_swap[label] for label in labels]
counts = []
cluster_data = []
for i, label in enumerate(unique_labels):
    count = np.sum([1 if lab == label else 0 for lab in labels])
    cd = {
        "label": label,
        "count": count,
        "disp": sum_euclid_dist_each[dict_swap[i]] / count,
        "centroid": list(centroids[dict_swap[i]])
    }
    cluster_data.append(cd)
print(cluster_data)
# quit()
# format results
res_data = "label,count,disp,duration,volume\n"
for cd in cluster_data:
    res_data += str(cd["label"]) + "," + str(cd["count"]) + "," + str(cd["disp"]
                                                                      ) + "," + str(cd["centroid"][0]) + "," + str(cd["centroid"][1]) + "\n"
# print(centroids)
if savefile:
    result_name = root_data_folder + "/res_" + fname + ".csv"
    with open(result_name, "w") as f:
        f.write(res_data)
# quit()

# plot the results
fig = clustering.plot_data_with_clusters(
    X, kmeans, True, "duration [x60 s]", "volume [L]", False, (8, 6), 0, "Event Clustering - " + title_dict[selection])
# add 4 squares delimiter
# plt.axvline(x=np.min(X[:, 0] + (np.max(X[:, 0]) - np.min(X[:, 0])) / 2))
# plt.axhline(y=0)
plt.show()

result_name = "./figs/event_clusters_" + fname + "_" + str(nc) + "c"

if savefig:
    graph.save_figure(fig, result_name)

if not eval_nc:
    quit()

# quit()
r = range(2, 20)
silhouette_score_vect = []
WCSS_vect = []

optimal_number_of_clusters = 0
max_silhouette_score = 0

for i in r:
    # run the k-means algorithm with i clusters
    X, kmeans, centroids, average_silhouette_score, wcss, average_euclid_dist_mean, _ = clustering.clustering_kmeans(
        x, i, True)
    # save the results into a list
    silhouette_score_vect.append(average_silhouette_score)
    WCSS_vect.append(wcss)
    # check the optimal number of clusters by silhouette score
    if average_silhouette_score > max_silhouette_score:
        max_silhouette_score = average_silhouette_score
        optimal_number_of_clusters = i

# plot the results
print("optimal number of clusters: ", optimal_number_of_clusters)
fig = plt.figure(1, figsize=(8, 6))
plt.subplot(211)
plt.plot(r, silhouette_score_vect)
# plt.legend(["silhouette score"])
plt.xticks(r)
plt.grid()
clustering.set_disp("Clustering evaluation", "", "silhouette score")
plt.subplot(212)
plt.plot(r, WCSS_vect)
# plt.legend(["WCSS"])
plt.xticks(r)
plt.grid()
clustering.set_disp("", "k", "WCSS")
plt.show()
result_name = "./figs/event_clusters_" + fname + "_eval"

if savefig:
    graph.save_figure(fig, result_name)
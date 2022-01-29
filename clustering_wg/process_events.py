

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np
import matplotlib.pyplot as plt


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
end_col = None
fill_start = False

x, header = loader.load_dataset(result_name)
df = loader.load_dataset_pd(result_name)

filter_labels = []
# filter_labels = ["toaleta", "chiuveta_rece", "chiuveta_calda"]
# filter_labels = ["toaleta"]


if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]
    # df = df[df['label'] == filter_labels]

print(df)
x = df.to_numpy()
x = x[start_index:, start_col:]

print(x)
print(np.shape(x))
# quit()

# run the k-means algorithm with nc (number of continents) clusters
X, kmeans, centroids, average_silhouette_score, wcss, average_euclid_dist_mean = clustering.clustering_kmeans(x, 3, True)

# plot the results
clustering.plot_data_with_clusters(X, kmeans, True, "duration [x60 s]", "volume [L]", False)
# add 4 squares delimiter
# plt.axvline(x=np.min(X[:, 0] + (np.max(X[:, 0]) - np.min(X[:, 0])) / 2))
# plt.axhline(y=0)
plt.show()


quit()
r = range(2, 20)
silhouette_score_vect = []
WCSS_vect = []

optimal_number_of_clusters = 0
max_silhouette_score = 0

for i in r:
    # run the k-means algorithm with i clusters
    X, kmeans, centroids, average_silhouette_score, wcss, average_euclid_dist_mean = clustering.clustering_kmeans(x, i, True)
    # save the results into a list
    silhouette_score_vect.append(average_silhouette_score)
    WCSS_vect.append(wcss)
    # check the optimal number of clusters by silhouette score
    if average_silhouette_score > max_silhouette_score:
        max_silhouette_score = average_silhouette_score
        optimal_number_of_clusters = i

# plot the results
print("optimal number of clusters: ", optimal_number_of_clusters)
plt.subplot(211)
plt.plot(r, silhouette_score_vect)
plt.title("Clustering evaluation")
plt.legend(["silhouette score"])
plt.xticks(r)
plt.subplot(212)
plt.plot(r, WCSS_vect)
plt.legend(["WCSS"])
plt.xticks(r)
plt.xlabel("k")
plt.show()

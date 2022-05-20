

# import our modules
from modules import loader, graph
from modules import clustering
from modules import utils
from modules import preprocessing
import numpy as np

root_data_folder = "./data"
# read the data from the csv file

filenames = ["data_consumer_types.csv"]


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
        # graph.save_figure(fig, "./figs/eval_trends_inertia.png")
        X, kmeans, centroids, silhouette_score, _, _ = clustering.clustering_kmeans(
            x, nc, True)
        print("optimal number of clusters: " + str(nc) +
              " (" + str(max_silhouette_score) + ")")
    else:
        X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean = clustering.clustering_kmeans(
            x, nc, True)
        # X, kmeans, centroids, avg_dist, sum_dist, average_euclid_dist_mean = clustering.clustering_birch(x, nc, True)

    print("silhouette score: ", silhouette_score)
    xc = np.transpose(centroids)
    # xc = np.transpose(xc)

    if xlabels is not None:
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

    sx = np.shape(xc)
    print(sx)
    # xc = xc[0:100, :]
    tss = utils.create_timeseries(xc, xheader, xlabels)
    return tss, nc


options = [

    {
        "nc": 4,
        "norm_sum": False,
        "norm_axis": False
    }
]

plot_all_data = True
plot_all_data = False

remove_outlier = True
remove_outlier = False

norm2 = True
norm2 = False

rolling_filter = True

for option in options:
    print(option)
    nc = option["nc"]

    if nc is None:
        nc_orig = "auto"
    else:
        nc_orig = nc

    # continue

    # create separate models for each data file
    for filename in filenames:
        data_file = root_data_folder + "/" + filename
        x, header = loader.load_dataset(data_file)    
        df = loader.load_dataset_pd(data_file, True)
        print(df)
        df = df.drop(['uid', 'label', 'x'], axis=1)
        x = df.to_numpy()        
        # x = np.nan_to_num(x)

        # check max dim
        sx = np.shape(x)
        x = preprocessing.remove_fit_max_cols(x)
        x = preprocessing.imputation(x)

        sx = np.shape(x)
        if rolling_filter:
            kernel_size = int(0.1 * sx[0])
            kernel = np.ones(kernel_size) / kernel_size
            for dim in range(sx[1]):
                x[:, dim] = np.convolve(x[:, dim], kernel, mode='same')

        if norm2:
            x = preprocessing.normalize(x)

        if remove_outlier:
            x = clustering.remove_outliers(x)

        # print(x)

        # quit()
        sx = np.shape(x)
        print(sx)
        print("start")

        # quit()

        # time axis labels
        xlabels = [str(i) for i in range(sx[1])]
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)
        print(xlabels)

        if plot_all_data:
            title = filename
            xplot = np.transpose(x)
            tss = utils.create_timeseries(xplot, None, None)
            fig = graph.plot_timeseries_multi_sub2(
                [tss], [title], "x", ["y"], None, None, 24)

        # cluster labels
        xheader = ["c" + str(i+1) for i in range(sx[1])]
        print(xheader)

        if nc is None:
            xlabels = [xlabels] * 1000
        else:
            xlabels = [xlabels] * nc

        # print(np.shape(xlabels))

        # quit()

        tss, nc = run_clustering(x, nc, xheader, xlabels)

        # plot cluster centroids
        title = filename

        title = "consumer patterns (" + str(nc) + "c)"

        ylabel = "y [L]"
        if norm2:
            ylabel = "y [norm]"

        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "x [samples]", [ylabel], None, 5, None)

        result_name = "./figs/consumer_patterns_" + str(nc_orig) + "c"
        if norm2:
            result_name += "_norm"

        graph.save_figure(fig, result_name)

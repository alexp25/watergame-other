

# import our modules
from modules import loader, graph
from modules import clustering
from modules import utils
from modules import preprocessing
from matplotlib import pyplot as plt


import numpy as np
import pandas as pd
import time
import os
import yaml
from typing import List
import math
import sys


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
        r = range(2,20)
        for nc1 in r:
            X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean = clustering.clustering_kmeans(x, nc1, True)
            silhouette_score_vect.append(silhouette_score)
            WCSS_vect.append(WCSS)
            # WCSS_vect.append(average_euclid_dist_mean)
            if silhouette_score > max_silhouette_score:
                max_silhouette_score = silhouette_score
                optimal_number_of_clusters = nc1
        nc = optimal_number_of_clusters
        fig = graph.plot(silhouette_score_vect, list(r), "Optimal number of clusters", "Number of clusters", "Silhouette score", True)
        WCSS_vect = utils.normalize_axis_01(np.array([WCSS_vect]), 1).tolist()[0]
        fig = graph.plot(WCSS_vect, list(r), "Optimal number of clusters", "Number of clusters", "WCSS", True)
        # graph.save_figure(fig, "./figs/eval_trends_inertia.png")
        X, kmeans, centroids, silhouette_score, _, _ = clustering.clustering_kmeans(x, nc, True)
        print("optimal number of clusters: " + str(nc) + " (" + str(max_silhouette_score) + ")")
    else:
        X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean = clustering.clustering_kmeans(x, nc, True)
        # X, kmeans, centroids, avg_dist, sum_dist, average_euclid_dist_mean = clustering.clustering_birch(x, nc, True)        

    print("silhouette score: ", silhouette_score)
    xc = np.transpose(centroids)
    # xc = np.transpose(xc)   

    if xlabels is not None:
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

    sx = np.shape(xc)
    print(sx)
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
# norm2 = False

for option in options:

    print(option)

    start_index = 1
    end_index = None
    start_col = 3
    end_col = 61
    fill_start = False
        
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
        # x = np.nan_to_num(x)
        nheader = len(header)

        x = preprocessing.imputation(x)

        if norm2:
            x = preprocessing.normalize(x)

        sx = np.shape(x)

        if fill_start:
            x = x[start_index:, :]
            x[:, 0:start_col-1] = np.transpose(np.array([[0] * (sx[0]-1)]))
        else:
            x = x[start_index:, start_col:]

        if end_index is not None:
            x = x[:end_index, :]
        if end_col is not None:
            x = x[:, :end_col]

        sx = np.shape(x)
        print(sx)

        if remove_outlier:
            outlier = -1
            outliers = []

            x = clustering.remove_outliers(x)

            # for i in range(sx[0]):
            #     if x[i,3] > 100:
            #         outliers.append(i)

            # print(outliers)

            # x = np.delete(x, obj=outliers, axis=0)

        # print(x)

        # quit()
        sx = np.shape(x)
        print(sx)

        print("start")

        # quit()

        header = []
        for d in range(nheader-1):
            header.append(str(d+1))

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
            [tss], [title], "x [months]", [ylabel], None, 5, None)

        result_name = "./figs/consumer_patterns_" + str(nc_orig) + "c"
        if norm2:
            result_name += "_norm"
            
        graph.save_figure(fig, result_name)

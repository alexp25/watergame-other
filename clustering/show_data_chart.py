

# import our modules
from modules import loader, graph
from modules import clustering
from modules import utils


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

filenames = ["Water weekly/water_avg_weekly.csv"]
# filenames = ["Smart Water Meter/processed/avg_out.csv"]
# filenames = ["Smart Water Meter/processed/avg_out_2t.csv"]


def run_clustering(x, nc, xheader, xlabels=None):
    # x = np.transpose(x)

    # sx = np.shape(x)
    # print(sx)

    if nc is None:
        # use silhouette score
        max_silhouette_score = 0
        silhouette_score_vect = []
        WCSS_vect = []
        optimal_number_of_clusters = 2
        r = range(2,11)
        for nc1 in r:
            X, kmeans, silhouette_score, WCSS = clustering.clustering_kmeans(x, nc1, True)
            silhouette_score_vect.append(silhouette_score)
            WCSS_vect.append(WCSS)
            if silhouette_score > max_silhouette_score:
                max_silhouette_score = silhouette_score
                optimal_number_of_clusters = nc1
        nc = optimal_number_of_clusters
        graph.plot(silhouette_score_vect, list(r))
        graph.plot(WCSS_vect, list(r))
        X, kmeans, silhouette_score, _ = clustering.clustering_kmeans(x, nc, True)
        print("optimal number of clusters: " + str(nc) + " (" + str(max_silhouette_score) + ")")
    else:
        X, kmeans, _, _ = clustering.clustering_kmeans(x, nc, True)
    xc = np.transpose(kmeans.cluster_centers_)
    # xc = np.transpose(xc)
    # print(xc)

    if xlabels is not None:
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

    sx = np.shape(xc)
    print(sx)
    tss = utils.create_timeseries(xc, xheader, xlabels)
    return tss, nc


options = [
    
    {
        "nc": 5,
        "norm_sum": False,
        "norm_axis": False
    },
    {
        "nc": 5,
        "norm_sum": True,
        "norm_axis": False
    },
    {
        "nc": 5,
        "norm_sum": False,
        "norm_axis": True
    },
    {
        "nc": 5,
        "norm_sum": True,
        "norm_axis": True
    },
    {
        "nc": 3,
        "norm_sum": False,
        "norm_axis": False
    },
    {
        "nc": 3,
        "norm_sum": True,
        "norm_axis": False
    },
    {
        "nc": 3,
        "norm_sum": False,
        "norm_axis": True
    },
    {
        "nc": 3,
        "norm_sum": True,
        "norm_axis": True
    },

    # {
    #     "nc": None,
    #     "norm_sum": False,
    #     "norm_axis": False
    # },
    # {
    #     "nc": None,
    #     "norm_sum": True,
    #     "norm_axis": False
    # },
    # {
    #     "nc": None,
    #     "norm_sum": False,
    #     "norm_axis": True
    # },
    # {
    #     "nc": None,
    #     "norm_sum": True,
    #     "norm_axis": True
    # },
]

plot_all_data = True
plot_all_data = False

remove_outlier = True
remove_outlier = False

norm = True
# norm = False

for option in options:

    print(option)

    start_index = 1
    end_index = None
    start_col = 6
    end_col = None
    fill_start = True

    norm_sum = option["norm_sum"]
    norm_axis = option["norm_axis"]
    norm = True

    if not norm_sum and not norm_axis:
        norm = False
        
    # print(norm_sum, norm)

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

        # quit()

        if norm:
            if norm_sum:
                x = utils.normalize_sum_axis(x, 0)
            if norm_axis:
                x = utils.normalize_axis_01(x, 1)

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

        tss, nc = run_clustering(x, nc, xheader, xlabels)

        # plot cluster centroids
        title = filename

        title = "weekly consumer patterns (" + str(nc) + "c)"

        ylabel = "y [L/h]"
        if norm:
            ylabel = "y [norm]"

        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "x [hours]", [ylabel], None, None, 24)

        result_name = "./figs/consumer_patterns_" + str(nc_orig) + "c"
        if norm:
            result_name += "_norm"
            if norm_sum:
                result_name += "_norm_sum"
            if norm_axis:
                result_name += "_norm_axis"
            
        graph.save_figure(fig, result_name)

        # run_clustering(y, times, yheader)
        # plot_data(x, y, xheader, yheader)

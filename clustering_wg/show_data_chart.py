

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils

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


# load sensors list
data_file = root_data_folder + "/" + "setup.csv"
df = loader.load_dataset_pd(data_file)

sensor_list = []
for row in df.iterrows():
    rowspec = row[1]
    if not np.isnan(rowspec["id"]):
        sensor_spec = {
            "id": int(rowspec["id"]),
            "labels": []
        }
        data_labels = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
        for dl in data_labels:
            try:
                if np.isnan(rowspec[dl]):
                    pass    
            except:
                sensor_spec["labels"].append(rowspec[dl])   
                pass    
        # print(row)
        sensor_list.append(sensor_spec)
        print(sensor_spec)


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
        graph.save_figure(fig, "./figs/eval_trends_inertia.png")
        X, kmeans, centroids, silhouette_score, _, _ = clustering.clustering_kmeans(x, nc, True)
        print("optimal number of clusters: " + str(nc) + " (" + str(max_silhouette_score) + ")")
    else:
        X, kmeans, centroids, silhouette_score, WCSS, average_euclid_dist_mean = clustering.clustering_kmeans(x, nc, True)
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


options = [
    
    {
        "nc": 4,
        "norm_sum": False,
        "norm_axis": False
    },
    # {
    #     "nc": 4,
    #     "norm_sum": True,
    #     "norm_axis": False
    # },
    # {
    #     "nc": 4,
    #     "norm_sum": False,
    #     "norm_axis": True
    # },
    # {
    #     "nc": 4,
    #     "norm_sum": True,
    #     "norm_axis": True
    # },

    # {
    #     "nc": 3,
    #     "norm_sum": False,
    #     "norm_axis": False
    # },
    # {
    #     "nc": 3,
    #     "norm_sum": True,
    #     "norm_axis": False
    # },
    # {
    #     "nc": 3,
    #     "norm_sum": False,
    #     "norm_axis": True
    # },
    # {
    #     "nc": 3,
    #     "norm_sum": True,
    #     "norm_axis": True
    # },

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
# plot_all_data = False

remove_outlier = True
remove_outlier = False

# norm = True
norm = False

extract_inst_flow = False
extract_inst_flow = True

rolling_filter = True
# rolling_filter = False

for option in options:

    print(option)

    start_index = 1
    # end_index = 100
    end_index = None
    start_col = 3
    end_col = None
    fill_start = False

    norm_sum = option["norm_sum"]
    norm_axis = option["norm_axis"]
    norm = True

    if not norm_sum and not norm_axis:
        norm = False

    nc = option["nc"]

    if nc is None:
        nc_orig = "auto"
    else:
        nc_orig = nc

    # create separate models for each data file
    for sensor_spec in sensor_list:
        filename = "watergame_sample_consumer_" + str(sensor_spec["id"]) + ".csv"
        data_file = root_data_folder + "/" + filename
        x, header = loader.load_dataset(data_file)

        df = loader.load_dataset_pd(data_file)
        # timestamps = df[["timestamp"]]
        timestamps = df["timestamp"]
        sid = df["sensorId"]
        # print(list(timestamps))
        # quit()
        
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
            x = x[:,:end_col]

        xlabel = ""
        xtype = ""

        if extract_inst_flow:
            xlabel = "flow [L/h]"
            xtype = "flow"
            title = "consumption data (flow) node " + str(sid[0])
            x = x[:,1::2]
        else:
            xlabel = "volume [L]"
            xtype = "volume"
            title = "consumption data (volume) node " + str(sid[0])
            x = x[:,0::2] / 1000

        sx = np.shape(x)
        print(sx)

        if rolling_filter:
            kernel_size = int(0.1 * sx[0])
            kernel = np.ones(kernel_size) / kernel_size
            for dim in range(sx[1]):
                x[:,dim] = np.convolve(x[:,dim], kernel, mode='same')


        if remove_outlier:
            outlier = -1
            outliers = []
            x = clustering.remove_outliers(x)

        if norm:
            if norm_sum:
                x = utils.normalize_sum_axis(x, 0)
            if norm_axis:
                x = utils.normalize_axis_01(x, 1)

        sx = np.shape(x)
        print(sx)

        print("start")

        header = []
        for d in range(nheader-1):
            header.append(str(d+1))
        header = sensor_spec["labels"]

        # time axis labels
        # xlabels = [str(i) for i in range(sx[1])]
        xlabels = [str(i) for i in range(len(timestamps))]
        xlabels = timestamps
        xlabels = [np.datetime64(ts) for ts in timestamps]
        # xlabels = np.array(xlabels)
        # xlabels = np.transpose(xlabels)
        # print(xlabels)

        if plot_all_data:
            xplot = x
            tss = utils.create_timeseries(xplot, header, None)
            fig = graph.plot_timeseries_multi_sub2(
                [tss], [title], "time", [xlabel], None, None, None, xlabels)

        result_name = "./figs/consumer_data_" + xtype + "_"
        if rolling_filter:
            result_name += "rf_"
        result_name += str(sid[0])
        graph.save_figure(fig, result_name)
        
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

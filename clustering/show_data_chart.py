

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
    X, kmeans, _, _ = clustering.clustering_kmeans(x, nc)
    xc = np.transpose(kmeans.cluster_centers_)
    # xc = np.transpose(xc)
    # print(xc)

    if xlabels is not None:
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

    sx = np.shape(xc)
    print(sx)
    tss = utils.create_timeseries(xc, xheader, xlabels)
    return tss


plot_all_data = True
plot_all_data = False
remove_outlier = True
remove_outlier = False
norm = True
norm = False

start_index = 1
end_index = None
start_col = 6
end_col = None
fill_start = True


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
        x = utils.normalize(x, 0)

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

    nc = 5
    # cluster labels
    xheader = ["c" + str(i) for i in range(sx[1])]
    print(xheader)

    xlabels = [xlabels]*nc

    tss = run_clustering(x, nc, xheader, xlabels)

    # plot cluster centroids
    title = filename
    fig = graph.plot_timeseries_multi_sub2(
        [tss], [title], "x", ["y"], None, None, 24)

    # graph.save_figure(fig, "./figs/consumer_patterns_all_2")

    # run_clustering(y, times, yheader)
    # plot_data(x, y, xheader, yheader)

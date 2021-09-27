

# import our modules
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

filenames = ["data_consumer_types.csv"]

def run_clustering(x, nc, xheader, xlabels=None):
    
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

norm = True
# norm = False

for option in options:

    print(option)

    start_index = 1
    end_index = None
    start_col = 3
    end_col = 61
    fill_start = False

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
        df = loader.load_dataset_pd(data_file)
        classes = df
        groups = classes.groupby(["Consumer"]).cumcount()
        x_groups_list = (df.set_index(['Consumer',groups])
            .unstack(fill_value=0)
            .stack().groupby(level=0)
            .apply(lambda x: x.values.tolist())
            .tolist())
        x_groups = np.array(x_groups_list)
        
        print(x_groups_list[0][1])

        quit()
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


        if norm:
            if norm_sum:
                x = utils.normalize_sum_axis(x, 0)
            if norm_axis:
                x = utils.normalize_axis_01(x, 1)

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

        title = "monthly consumer patterns (" + str(nc) + "c)"

        ylabel = "y [L]"
        if norm:
            ylabel = "y [norm]"

        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "x [months]", [ylabel], None, 5, None)

        result_name = "./figs/consumer_patterns_" + str(nc_orig) + "c"
        if norm:
            result_name += "_norm"
            if norm_sum:
                result_name += "_norm_sum"
            if norm_axis:
                result_name += "_norm_axis"
            
        # graph.save_figure(fig, result_name)

        # run_clustering(y, times, yheader)
        # plot_data(x, y, xheader, yheader)

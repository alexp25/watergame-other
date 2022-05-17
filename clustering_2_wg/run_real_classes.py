

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
filenames_aux = ["data_consumer_types_content.csv"]


def run_clustering(x, nc, xheader, xlabels=None):
    xc = np.transpose(x)
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

norm2 = True
# norm2 = False

for option in options:

    print(option)

    start_index = 1
    end_index = None
    start_col = 3
    end_col = 61
    fill_start = False

    # print(norm_sum, norm)

    nc = option["nc"]

    if nc is None:
        nc_orig = "auto"
    else:
        nc_orig = nc

    # continue

    # create separate models for each data file
    for i, filename in enumerate(filenames):
        data_file = root_data_folder + "/" + filename
        data_file_aux = root_data_folder + "/" + filenames_aux[i]
        x, header = loader.load_dataset(data_file)
        x2, _ = loader.load_dataset(data_file_aux)
        x2 = preprocessing.imputation(x2)

        if norm2:
            x2 = preprocessing.normalize(x)

        df = loader.load_dataset_pd(data_file)

        print("\n\n\n")

        classes = df

        groups = classes.groupby(["Consumer"]).groups
        groups_dict = {}

        for group in groups:
            idx = list(groups[group])
            key = str(group)
            groups_dict[key] = {
                "group": key,
                "idx": idx,
                "dataset": [],
                "mean": None
            }
            dataset = []
            for i in idx:
                dataset.append(x2[i])
            groups_dict[key]["dataset"] = np.array(dataset)
            groups_dict[key]["mean"] = np.average(dataset, axis=0)
            # groups_dict[key]["mean"] = np.array(dataset[12])

        x_groups = []
        for k in groups_dict:
            gd = groups_dict[k]
            print(len(gd["idx"]))
            print(np.shape(gd["mean"]))
            x_groups.append(gd["mean"])

        x_groups = np.array(x_groups)
        print(np.shape(x_groups))
        print(x_groups)

        # x = x_groups[2:,:]
        x = x_groups

        start_index = 0
        end_index = None
        start_col = 3
        end_col = 61
        fill_start = False

        if fill_start:
            x = x[start_index:, :]
            x[:, 0:start_col-1] = np.transpose(np.array([[0] * (sx[0]-1)]))
        else:
            x = x[start_index:, start_col:]

        if end_index is not None:
            x = x[:end_index, :]
        if end_col is not None:
            x = x[:, :end_col]

        nheader = len(header)
        sx = np.shape(x)
        print(sx)
        print("start")

        # quit()

        header = []
        for d in range(nheader-1):
            header.append(str(d+1))
        # header = ["multi-family", "multi-family - irrigation", "residential", "residential - irrigation"]

        # time axis labels
        xlabels = [str(i) for i in range(sx[1]+1)]
        
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)

        if nc is None:
            xlabels = [xlabels] * 1000
        else:
            xlabels = [xlabels] * nc

        title = "consumer patterns / type (" + str(nc) + "c)"
        ylabel = "y [L]"
        if norm2:
            ylabel = "y [norm]"
            
        xplot = np.transpose(x)
        if xlabels is not None:
            xlabels = np.array(xlabels)
            xlabels = np.transpose(xlabels)

        print(xlabels)
        print(np.shape(xplot))

        # quit()
        # quit()

        tss = utils.create_timeseries(xplot, header, xlabels)

        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "x [months]", [ylabel], None, 5, None)

        result_name = "./figs/consumer_types_" + str(nc_orig) + "c"
        if norm2:
            result_name += "_norm"
        graph.save_figure(fig, result_name)


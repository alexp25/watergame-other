

# import our modules
from modules import loader, graph
from modules import clustering
from modules import utils
from modules import clustering_eval, clustering_eval_2
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
    assignments = clustering.clustering_kmeans_get_labels(x, nc, True)
    return assignments


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

    # create separate models for each data file
    for filename in filenames:
        data_file = root_data_folder + "/" + filename
        x, header = loader.load_dataset(data_file)
        df = loader.load_dataset_pd(data_file)
        classes = df[["ID", "Consumer"]]
        # print(classes)
        nheader = len(header)
        x = preprocessing.imputation(x)
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

        if norm2:
            x = preprocessing.normalize(x)

        sx = np.shape(x)
        print(sx)
        print("start")

        header = []
        for d in range(nheader-1):
            header.append(str(d+1))

        # time axis labels
        xlabels = [str(i) for i in range(sx[1])]
        xlabels = np.array(xlabels)
        xlabels = np.transpose(xlabels)
        print(xlabels)


        # cluster labels
        xheader = ["c" + str(i+1) for i in range(sx[1])]
        print(xheader)

        if nc is None:
            xlabels = [xlabels] * 1000
        else:
            xlabels = [xlabels] * nc

        classes_ident = run_clustering(x, nc, xheader, xlabels)
        classes["Cluster"] = classes_ident + 1

        classes.to_csv('out.csv', index=False)

        print(classes)
        xc = np.array([classes["Consumer"].to_list(),
                       classes["Cluster"].to_list()])
        xc = np.transpose(xc)

        print(xc)
        eval_data =  [classes["Consumer"].to_list(), classes["Cluster"].to_list()]
        clustering_eval.eval_rand_index(eval_data[0], eval_data[1])
        clustering_eval.eval_purity(eval_data[0], eval_data[1])
        p = clustering_eval_2.purity(eval_data[0], eval_data[1])
        print("purity: ", p)
        p = clustering_eval_2.entropy(eval_data[0], eval_data[1])
        print("entropy: ", p)
        p = clustering_eval_2.rand_index(eval_data[0], eval_data[1])
        print("rand index: ", p)
        p = clustering_eval_2.adj_rand_index(eval_data[0], eval_data[1])
        print("adj rand index: ", p)
        quit()

        # quit()
        xheader = ["Consumer Type", "Cluster"]
        xlabels = np.array([classes["ID"], classes["ID"]])
        xlabels = np.transpose(xlabels)
        tss = utils.create_timeseries(xc, xheader, xlabels)

        # # plot cluster centroids
        title = filename
        title = "Consumer Types vs Clustering Results"
        ylabel = "Class"

        print(tss)

        # quit()

        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "Consumer ID", [ylabel], None, 10, None)

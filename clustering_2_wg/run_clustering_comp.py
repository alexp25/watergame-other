

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
import apply_filters


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

filter_labels = ["toaleta", "chiuveta_rece", "chiuveta_calda", "dus"]

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

    # create separate models for each data file
    for filename in filenames:
        data_file = root_data_folder + "/" + filename
        df = loader.load_dataset_pd(data_file, False)
        print(df)
        df = apply_filters.apply_filter_labels(df, filter_labels) 
        df = loader.format_data(df)
        print(df)
        # quit()

        classes = df[["uid", "label"]]
        classes["Consumer"] = classes["label"]
        print(classes)
    
        df = df.drop(['uid', 'label', 'x'], axis=1)
        x = df.to_numpy()
        sx = np.shape(x)
        x = preprocessing.remove_fit_max_cols(x)
        x = preprocessing.imputation(x)        

        sx = np.shape(x)
        print(sx)

        if rolling_filter:
            kernel_size = int(0.1 * sx[0])
            kernel = np.ones(kernel_size) / kernel_size
            for dim in range(sx[1]):
                x[:, dim] = np.convolve(x[:, dim], kernel, mode='same')

        if remove_outlier:
            outlier = -1
            outliers = []
            x = clustering.remove_outliers(x)

        if norm2:
            x = preprocessing.normalize(x)

        sx = np.shape(x)
        print(sx)
        print("start")

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
        eval_data = [classes["Consumer"].to_list(), classes["Cluster"].to_list()]
 
        clustering_eval.eval_rand_index(eval_data[0], eval_data[1])
        # clustering_eval.eval_purity(eval_data[0], eval_data[1])
        p = clustering_eval_2.purity(eval_data[0], eval_data[1])
        print("purity: ", p)
        p = clustering_eval_2.entropy(eval_data[0], eval_data[1])
        print("entropy: ", p)
        p = clustering_eval_2.rand_index(eval_data[0], eval_data[1])
        print("rand index: ", p)
        p = clustering_eval_2.adj_rand_index(eval_data[0], eval_data[1])
        print("adj rand index: ", p)
        # quit()

        # quit()
        xheader = ["Consumer Type", "Cluster"]
        xlabels = np.array([classes["uid"], classes["uid"]])
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

import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import loader, graph
from modules.graph import Timeseries
import time
import os
import yaml
from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]
filenames = config["filenames"]


def create_timeseries(data, header):
    tss: List[Timeseries] = []
    # colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']

    ck = 0

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    colors = cm.rainbow(np.linspace(0, 1, cols))
    # colors = cm.viridis(np.linspace(0, 1, cols))

    for j in range(cols):
        ts: Timeseries = Timeseries()
        ts.label = header[j]
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        for i in range(rows):
            ts.x.append(i)
            ts.y.append(data[i][j])

        tss.append(ts)
        ts = None

    return tss


# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, xheader, yheader = loader.load_dataset(data_file, True)

    print(xheader)
    print(yheader)
    print(len(xheader))

    print(x)

    print(np.shape(x))
    print(np.shape(y))

    quit()

    # y = remove_outliers(y, 100)
    tss = create_timeseries(y, yheader)
    # x = remove_outliers(x)
    tss2 = create_timeseries(x, xheader)
    fig, _ = graph.plot_timeseries_multi_sub2([tss, tss2], [
                                              "valve sequence", "sensor output"], "samples [x0.1s]", ["position [%]", "flow [L/h]"])

    graph.save_figure(fig, "./figs/data_" + filename)

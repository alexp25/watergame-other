

# import our modules
from modules import loader, graph
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
filename = "data_sorted.csv"

data_file = root_data_folder + "/" + filename
x, header = loader.load_dataset(data_file)
# x = np.nan_to_num(x)
nheader = len(header)
x = x/1000
sx = np.shape(x)
print(sx)

# time axis labels
xlabels = [str(i) for i in range(sx[1])]
xlabels = np.array(xlabels)
xlabels = np.transpose(xlabels)
print(xlabels)

header = ["100 threads", "100 threads w/ REDIS", "100 threads / no optimization"]
header = ["no optimization", "query optimization", "w/ REDIS"]

title = filename
title = "System benchmark"
# xplot = np.transpose(x)
xplot = x
tss = utils.create_timeseries(xplot, header, None)
fig = graph.plot_timeseries_multi_sub2(
    [tss], [title], "progress (requests sent)", ["response time (s)"], None, None, None)
    
graph.save_figure(fig, "results.png")

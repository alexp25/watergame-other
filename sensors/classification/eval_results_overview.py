from modules import graph

from modules.graph import Timeseries, Barseries

from os import listdir
from os.path import isfile, join
import json
from typing import List
import numpy as np
import matplotlib
import yaml
import csv

import math


with open("config.yml", "r") as f:
    config = yaml.load(f)

labels = ["DT", "RF", "Dense", "RNN"]
limits = [75, 100]
files_array = [["eval_dtree_train", "eval_randomforest_train",
                "eval_dense_train", "eval_rnn_train"], ["eval_dtree_test", "eval_randomforest_test",
                                                        "eval_dense_test", "eval_rnn_test"]]
scale = [1, 1, 100, 100]
acc = []
counter = 0
for files in files_array:
    acc.append([])
    for i, input_file in enumerate(files):
        input_file = "./output/" + input_file + ".csv"
        with open(input_file, "r") as f:
            content = f.read().split("\n")
            line = content[0]
            spec = line.split(",")
            acc1 = float(spec[0]) * scale[i]
            acc[counter].append(acc1)
    counter += 1

print(acc)
print("create barseries")

print("plotting chart")
fig = graph.plot_barchart(
    labels, acc[1], "model", "accuracy [%]", "Classification accuracy", None, limits)
graph.save_figure(fig, "./figs/eval_accuracy_comp")

print(len(files))
cmap = matplotlib.cm.get_cmap('viridis')
color_scheme = [cmap(i) for i in np.linspace(0, 1, len(files))]
print(color_scheme)

fig, _ = graph.plot_barchart_multi_core_raw(acc, color_scheme, ["train", "test"], "model", "accuracy [%]",
                                            "Classification accuracy", ["Decision Tree",
                                                     "Random Forest", "Dense", "RNN"],
                                            limits,
                                            True, None, 0, None)
graph.save_figure(fig, "./figs/eval_accuracy_comp_dual")

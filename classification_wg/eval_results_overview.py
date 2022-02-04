from modules import graph

from modules.graph import Timeseries, Barseries

from os import listdir
from os.path import isfile, join
import json
from typing import List
import numpy as np

import yaml
import csv

import math


with open("config.yml", "r") as f:
    config = yaml.load(f)

labels = ["DT", "RF", "Dense", "RNN"]
limits = [0, 100]
files = ["eval_dtree_test", "eval_randomforest_test",
         "eval_dense_test", "eval_rnn_test"]
scale = [1, 1, 100, 100]
acc = []
for i, input_file in enumerate(files):
    input_file = "./output/" + input_file + ".csv"
    with open(input_file, "r") as f:
        content = f.read().split("\n")
        line = content[0]
        spec = line.split(",")
        acc1 = float(spec[0]) * scale[i]
        acc.append(acc1)

print(acc)
# quit()
print("create barseries")

print("plotting chart")
fig = graph.plot_barchart(
    labels, acc, "model", "accuracy [%]", "Classification accuracy", None, limits)
graph.save_figure(fig, "./figs/eval_accuracy_comp")

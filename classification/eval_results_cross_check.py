from modules import graph

from modules.graph import Timeseries, CMapMatrixElement
import numpy as np

from os import listdir
from os.path import isfile, join
import json
from typing import List

import yaml

# import copy

with open("config.yml", "r") as f:
    config = yaml.load(f)

elements: List[CMapMatrixElement] = []

rowsdict = {}
colsdict = {}

mode = "deep_1"
mode = "deep_2_rnn"
# mode = "dtree_1"
# mode = "dtree_2_multioutput"
# mode = "svm_1"
# mode = "naive_bayes_1"

mode2 = "train"
mode2 = "test"

input_file = "./data/selected/output/cross_check_" + mode + "_" + mode2 + ".csv"

with open(input_file, "r") as f:
    content = f.read().split("\n")

    for line in content:
        spec = line.split(",")
        if len(spec) > 1:
            print(spec)
            e = CMapMatrixElement()
            e.i = int(spec[0])-1
            e.j = int(spec[1])-1
            e.val = float(spec[4])
            elements.append(e)

            if spec[0] not in rowsdict:
                rowsdict[spec[0]] = True
            if spec[0] not in colsdict:
                colsdict[spec[0]] = True

print(elements)

labels = ["1-N-80", "1-N-1-80", "1-N-1-50", "GRAY-80"]
labels = ["A", "B", "C", "ABC"]

xlabels = list(rowsdict)
ylabels = list(colsdict)

xlabels = ylabels = labels


xsize = len(rowsdict)
ysize = len(colsdict)

print(xsize)
print(ysize)

# intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
intersection_matrix = np.zeros((xsize, ysize))

# print(intersection_matrix)

avg = 0
count = 0

for e in elements:
    intersection_matrix[e.i][e.j] = e.val

    # if e.val > 0:
    avg += e.val
    count += 1

print(intersection_matrix)

avg /= count
# avg = np.mean(intersection_matrix)

print(avg)

# quit()

fig = graph.plot_matrix_cmap(elements, len(rowsdict), len(
    colsdict), "Model accuracy cross-validation (" + mode2 + ")", "dataset", "model", xlabels, ylabels, (70, 100))
graph.save_figure(fig, "./figs/accuracy_cross_check_" + mode + "_" + mode2)

from modules import graph

from modules.graph import Timeseries

from os import listdir
from os.path import isfile, join
import json
from typing import List

import yaml

# import copy

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_model_folder = config["root_model_folder"]

use_rnn = True
top_only = True

if config["load_from_container"]:
    if use_rnn:
        root_model_folder = config["root_model_container"] + "/deep_rnn"
    else:
        root_model_folder = config["root_model_container"] + "/deep"

mypath = root_model_folder


def list_files():

    if top_only:
        onlyfiles = [f for f in listdir(mypath) if isfile(
        join(mypath, f)) and '.txt' in f and "_top" in f]
    else:
        onlyfiles = [f for f in listdir(mypath) if isfile(
        join(mypath, f)) and '.txt' in f]
    return onlyfiles


files = list_files()

print(files)

def sort_by_exp(files):
    exp_files = {}
    for f in files:
        f1 = f.split("_")
        exp_id = f1[1]
        if exp_id in exp_files:
            exp_files[exp_id].append(f)
        else:
            exp_files[exp_id] = [f]
    return exp_files

# print(files)


sorted_files = sort_by_exp(files)

print(sorted_files)


def read_results(sorted_files):
    acc = {}
    for exp in sorted_files:
        # print(sorted_files[exp])
        for filename in sorted_files[exp]:
            with open(join(mypath, filename), "r") as f:
                accuracy = float(f.read().split("accuracy: ")
                                 [1].split("\n")[0])
                sorted_files[exp]
                if exp in acc:
                    acc[exp].append(accuracy)
                else:
                    acc[exp] = [accuracy]
    return acc


acc = read_results(sorted_files)

print(acc)


def create_timeseries(acc):
    tss: List[Timeseries] = []
    colors = ['blue', 'red', 'green', 'orange']
    ck = 0
    for (ik, key) in enumerate(acc):
        ts: Timeseries = Timeseries()
        ts.label = key
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0
        for (i, val) in enumerate(acc[key]):
            ts.x.append(i+1)
            ts.y.append(val)
        # print(len(ts.y))
        tss.append(ts)
        ts = None
    return tss


tss = create_timeseries(acc)
print(json.dumps(acc, indent=2))

labels = [key for key in acc]

labels = ["1-N-80%", "1-N-1-80%", "1-N-1-50%", "GRAY-80%"]

fig = graph.plot_barchart(labels, [acc[key][0]*100 for key in acc], "model", "accuracy [%]", "Max accuracy (deep learning)", "blue")

graph.save_figure(fig, "./figs/accuracy_models_comp")


# graph.plot_timeseries_multi(tss, "deep learning", "run", "accuracy", False)

# graph.stem_timeseries_multi(tss, "deep learning", "run", "accuracy", True)
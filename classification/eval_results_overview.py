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

# import copy

with open("config.yml", "r") as f:
    config = yaml.load(f)


mypath = "./data/output/"

# https://www.thinkingondata.com/something-about-viridis-library/
color_scheme = ['#404788FF', '#238A8DFF', '#FDE725FF', '#55C667FF']
color_scheme = ['#481567FF', '#2D708EFF', '#FDE725FF', '#55C667FF']

mode = "test"
# mode = "train"

models = ["deep_1", "deep_2_rnn", "dtree_1", "dtree_2_multioutput"]
labels = ["Dense", "RNN", "DT", "RF"]

# show model computation (only)

show_model_computation = False
# show_model_computation = True

def list_files(mode):
    onlyfiles = [f for f in listdir(mypath) if isfile(
        join(mypath, f)) and '.csv' in f and "eval_" in f and "_" + mode in f]
    return onlyfiles


def read_results(sorted_files):
    acc = {}
    for exp in sorted_files:
        with open(join(mypath, exp), "r") as f:
            data = f.read().split("\n")
            data2 = []
            for d in data:
                d = d.split(",")
                print(d)
                if len(d) > 1:
                    d_obj = {
                        "label": d[0],
                        "avg": float(d[1]),
                        "top": float(d[2]),
                        "dt": 0,
                        "fsize": 0
                    }

                    if len(d) > 4:
                        d_obj["dt"] = float(d[3])
                        d_obj["fsize"] = float(d[4])/1024

                    data2.append(d_obj)
            if len(data2) > 0:
                acc[exp] = data2
    return acc


if show_model_computation:
    files_train = list_files("train")
    acc_train = read_results(files_train)
    files_test = list_files("test")
    acc_test = read_results(files_test)

    for a in acc_train:
        for i, exp in enumerate(acc_train[a]):
            atrain = exp
            atest = acc_test[a.replace("_train", "_test")][i]
            atest["dt"] = atrain["dt"]

    # print(acc_train)
    # quit()

    # add train data (file size)
    acc = acc_test
else:
    files = list_files(mode)
    print(files)
    acc = read_results(files)

print(acc)

# quit()


def create_barseries(accs, keys, k):
    global color_scheme

    tss: List[Barseries] = []
    colors = ['blue', 'red', 'orange', 'green']
    colors = color_scheme

    ck = 0

    print("key: " + k)

    for (i, acc) in enumerate(accs):

        if i >= len(keys):
            break

        print(acc)

        ts: Barseries = Barseries()
        ts.label = keys[i]
        ts.color = colors[ck]

        ck += 1
        if ck >= len(colors):
            ck = 0

        ts.data = []

        acc_data = accs[acc]

        for (j, key) in enumerate(acc_data):
            # print(key)
            ts.data.append(key[k])

        print(ts.data)
        average_acc = np.mean(np.array(ts.data))

        print("average accuracy: ", average_acc)
        ts.average = acc + ": " + str(average_acc)

        tss.append(ts)

        ts = None
    return tss

# particular case


def create_barseries_avg_accuracy_for_model_interlaced(accs, avg_best_selector, models):
    global color_scheme

    tss: List[Barseries] = []
    colors = ['blue', 'red', 'green', 'orange']
    colors = color_scheme
    ck = 0

    for (i, avg_best) in enumerate(avg_best_selector):
        print("key: " + avg_best)
        ts: Barseries = Barseries()
        ts.label = avg_best
        ts.color = colors[ck]
        ts.data = []

        ck += 1
        if ck >= len(colors):
            ck = 0

        # accs <=> models
        for (j, model) in enumerate(accs):
            acc_data = accs[model]
            batch = []
            for (j2, data1) in enumerate(acc_data):
                batch.append(acc_data[j2][avg_best])

            ts.data.append(np.mean(np.array(batch)))

        tss.append(ts)
        ts = None
    return tss


print(acc)

# quit(0)

report = ""

keys_comp = ["avg", "top"]

limits = [80, 100]

if not show_model_computation:
    print("create barseries")
    tss = create_barseries_avg_accuracy_for_model_interlaced(
        acc, keys_comp, labels)
    print("plotting chart")

    fig = graph.plot_barchart_multi(
        tss, "model", "accuracy [%]", "Average model accuracy (" + mode + ")", labels, limits)

    graph.save_figure(fig, "./figs/eval_accuracy_comp_mean_combined_" + mode)

    print("\n\n")
    print("combined accuracy results: ")
    r = [str(ts.data) for ts in tss]
    report += "combined accuracy results:\n"
    report += "\n".join(r) + "\n\n"
    print(r)
    print("\n\n")

if show_model_computation:

    print("create barseries")

    # print(acc)
    keys_comp = ["avg"]

    tss = create_barseries_avg_accuracy_for_model_interlaced(
        acc, keys_comp, labels)

    keys_comp = ["dt"]
    tss1 = create_barseries_avg_accuracy_for_model_interlaced(
        acc, keys_comp, labels)
    keys_comp = ["fsize"]
    tss2 = create_barseries_avg_accuracy_for_model_interlaced(
        acc, keys_comp, labels)

    # for i, ts in enumerate(tss2):
    #     for j, d in enumerate(ts.data):
    #         ts.data[j] /= 1024
    # ts.data[j] = math.log(ts.data[j])

    # for i, ts in enumerate(tss1):
    #     for j, d in enumerate(ts.data):
    #         ts.data[j] /= 10

    print("plotting chart")
    r = [str(ts.data) for ts in tss1]
    print(r)
    r = [str(ts.data) for ts in tss2]
    print(r)

    r1 = [ts.data for ts in tss1][0]
    # r1 = [math.log(e) for e in r1]
    print(r1)
    r2 = [ts.data for ts in tss2][0]
    # r2 = [math.log(e) for e in r2]
    print(r2)

    # rads = [[int(max([10, math.log(d)*100]))
    #          for d in ts.data] for ts in tss][0]
    rads = [ts.data for ts in tss][0]
    accuracies = [int(d) for d in rads]

    min_acc = min(rads)
    max_acc = max(rads)

    rads = [max(0.1, (d-min_acc)/(max_acc-min_acc)) * 5000 for d in rads]
    print(rads)

    # quit()

    scalex = [50, 70000]
    scaley = [-30, 140]

    # scalex[0] = 2
    # scalex[1] = math.log(scalex[1])

    # scaley[0] = -3
    # scaley[1] = math.log(scaley[1])

    scale = [scalex, scaley]
    # scale = None

    fig = graph.plot_xy(r1, r2, rads, labels, color_scheme,
                  "Model evaluation", "size on disk [KB]",  "training time [s]", scale, True, accuracies)
    # quit()

    # fig = graph.plot_barchart_multi_dual(
    #     tss1, tss2, "model", "training time [x10 s]", "size on disk [MB]", "Model computation", labels, [[0, 12], [0, 30]], True)

    graph.save_figure(
        fig, "./figs/eval_accuracy_comp_mean_combined_aux_" + mode)

    quit()

print("create barseries")
tss = create_barseries(acc, labels, keys_comp[0])
print("plotting chart")
fig = graph.plot_barchart_multi(tss, "model", "accuracy [%]", "Average model accuracy (" + mode + ")", [
                                "1-N-80", "1-N-1-80", "1-N-1-50", "GRAY-80"], limits)
graph.save_figure(fig, "./figs/eval_accuracy_comp_mean_" + mode)

print("\n\n")
print("average accuracy results: ")
r = [ts.average for ts in tss]
report += "average accuracy results:\n"
report += "\n".join(r) + "\n\n"
print(r)
print("\n\n")

# print(keys_comp)

keys_comp = ["avg", "top"]

tss = create_barseries(acc, labels, keys_comp[1])
fig = graph.plot_barchart_multi(tss, "model", "accuracy [%]", "Best model accuracy (" + mode + ")", [
                                "1-N-80", "1-N-1-80", "1-N-1-50", "GRAY-80"], limits)
graph.save_figure(fig, "./figs/eval_accuracy_comp_best_" + mode)

print("\n\n")
print("top accuracy results: ")
r = [ts.average for ts in tss]
report += "top accuracy results:\n"
report += "\n".join(r) + "\n\n"
print(r)
print("\n\n")

with open(mypath + "accuracy_report.txt", "w") as f:
    f.write(report)

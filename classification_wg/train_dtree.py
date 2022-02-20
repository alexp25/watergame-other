import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import loader, model_loader, graph, print_tree
from modules.graph import Barseries
import time
import os
from shutil import copyfile, copy2
import yaml
import json
from modules.preprocessing import Preprocessing
from modules import preprocessing
from modules import generator
from typing import List

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]
filename = config["filename"]
modelname = config["modelname"]

method = "dtree"
method = "randomforest"

n_reps = 5
use_saved_model = False

append_timestamp = False
save_best_model = True

if use_saved_model:
    n_reps = 1

acc_train_vect = {}
acc_test_vect = {}

prep = Preprocessing()

if config["run_clean"] and not use_saved_model:
    loader.clean(root_model_folder)

# create separate models for each data file

data_file = root_data_folder + "/" + filename + ".csv"

df = loader.load_dataset_pd(data_file)
filter_labels = []
filter_labels = ["toaleta", "chiuveta_rece", "chiuveta_calda", "dus"]

if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]
    # df["duration"] > 0
    df = df[df["volume"] >= 1]
    # df = df[df["duration"] < 10]

df = loader.format_data(df)
print(df)

# shuffle dataset rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# this makes sure, that you keep your random choice always replicatable
# df.sample(n=len(df), random_state=42)
# where
# frac=1 means all rows of a dataframe
# random_state=42 means keeping same order in each execution
# reset_index(drop=True) means reinitialize index for randomized dataframe

X, y, features, classes = loader.get_dataset_xy(df)
X = X.to_numpy()
y = y.to_numpy()

print(X)
print(y)

# X = preprocessing.imputation(X)
# X = preprocessing.normalize(X)

acc_train_vect = {
    "data": [],
    "aux": [],
    "files": [],
    "acc": [],
    "avg": 0,
    "top": 0
}
acc_test_vect = {
    "data": [],
    "aux": [],
    "files": [],
    "acc": [],
    "avg": 0,
    "top": 0
}

top_acc = 0
top_model_filename = None

# session = K.get_session()

# classifiers.create_decision_tree(x, y[:,0], 20)
sizey = np.shape(y)

for rep in range(n_reps):
    model_file = root_model_folder + "/" + modelname
    model_file_raw = model_file
    model_file_raw += "_" + str(rep+1)

    if append_timestamp:
        app = "_" + str(time.time())
        model_file_raw += app

    model_file = model_file_raw + ".skl"

    n_train_percent = config["train_percent"]

    x_train, y_train = classifiers.split_dataset_train(
        X, y, n_train_percent)
    x_test, y_test = classifiers.split_dataset_test(
        X, y, n_train_percent)

    dt = 0

    if not use_saved_model:
        tstart = time.time()

        if method == "dtree":
            model = classifiers.create_decision_tree()
        elif method == "randomforest":
            model = classifiers.create_random_forest()

        # model = classifiers.create_svm()
        # model = classifiers.create_svm_multiclass()
        # model = classifiers.create_naive_bayes()
        model, acc = classifiers.train_decision_tree(
            model, x_train, y_train)
        dt = time.time() - tstart
    else:
        model = model_loader.load_sklearn_model(model_file)
        # print_tree.plot_decision_tree(model, features, classes, "dtree.png")

    model, acc, diff, total, _ = classifiers.predict_decision_tree(
        model, x_train, y_train, False)

    acc_train_vect["data"].append(diff)
    acc_train_vect["aux"].append(total)
    acc_train_vect["files"].append(model_file)
    acc_train_vect["acc"].append(acc)

    model, acc, diff, total, _ = classifiers.predict_decision_tree(
        model, x_test, y_test, False)

    model_loader.save_sklearn_model(model, model_file)
    acc_test_vect["data"].append(diff)
    acc_test_vect["aux"].append(total)
    acc_test_vect["files"].append(model_file)
    acc_test_vect["acc"].append(acc)


def set_avg(dc):
    # for each input file (experiment)
    acc_vect = np.array(dc["acc"])
    dc["avg"] = np.mean(acc_vect)
    dc["top"] = np.max(acc_vect)
    dc["best_model"] = dc["files"][np.argmax(acc_vect)]
    print(dc)


set_avg(acc_train_vect)
set_avg(acc_test_vect)

print("\ntrain vect")
print(json.dumps(acc_train_vect))
print("\ntest vect")
print(json.dumps(acc_test_vect))


def create_barseries(accs, keys):
    tss: List[Barseries] = []
    colors = ['blue', 'red', 'green', 'orange']
    ck = 0
    for (i, acc) in enumerate(accs):
        ts: Barseries = Barseries()
        ts.label = keys[i]
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        ts.data = []
        ts.data.append(acc)

        # print(ts.data)
        tss.append(ts)
        ts = None
    return tss


def extract_csv(vect):
    csvoutput = ""
    csvoutput = str(vect["avg"]) + "," + str(vect["top"]) + \
        "," + str(vect["best_model"]) + "\n"
    return csvoutput


csvoutput = extract_csv(acc_train_vect)

with open("./output/eval_" + method + "_train.csv", "w") as f:
    f.write(csvoutput)

csvoutput = extract_csv(acc_test_vect)

with open("./output/eval_" + method + "_test.csv", "w") as f:
    f.write(csvoutput)

print(acc_train_vect)
print(acc_test_vect["avg"])

# quit()
tss = create_barseries(
    [acc_test_vect["avg"], acc_test_vect["top"]], ["avg", "best"])
print(tss)
# quit()

fig = graph.plot_barchart_multi(
    tss, "model", "accuracy", "Average accuracy", ["Decision Tree"], [70, 100])

graph.save_figure(fig, "./figs/mean_accuracy_" + method)

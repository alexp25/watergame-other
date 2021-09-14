import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import loader, model_loader
from modules.preprocessing import Preprocessing
from modules import generator

import time 

from os import listdir
from os.path import isfile, join

import os
from shutil import copyfile, copy2

import yaml

features = []
# df = pd.read_csv(input_file, header=0)

# print(df)

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]
# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]

use_top = True
output_file = ""

if config["load_from_container"]:   
    root_model_folder = config["root_model_container"] + "/naive_bayes"
    output_file = "naive_bayes_1"
    

def list_files(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(
        join(mypath, f)) and ((use_top and "top" in f) or not use_top) and '.skl' in f]
    return onlyfiles

model_files = list_files(root_model_folder)
print(model_files)
print(filenames)

results_train = []
results_test = []

expnames = [name.split("exp_")[1] for name in filenames]

print(expnames)

# quit()

prep = Preprocessing()

if config["one_hot_encoding"]:
    prep.create_encoder(prep.adapt_input(
        generator.generate_binary(config["n_valves"])))

for (j, filename) in enumerate(filenames):
    # load dataset
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, _, _ = loader.load_dataset(data_file)
    y = loader.binarize(y)

    if config["one_hot_encoding"]:
        # y = prep.encode(prep.adapt_input(y))
        y = prep.encode(prep.adapt_input(y))
        y = prep.decode_int_onehot(y)

    expname = data_file.split("exp_")[1].split(".csv")[0]

    print(expname)

    print(np.shape(x))
    print(np.shape(y))

    for (k, model_file1) in enumerate(model_files):

        diff_sum_test = 0
        diff_sum_train = 0
        total_sum_test = 0
        total_sum_train = 0

        print(model_file1)

        # quit()
        # load model
        model_file = root_model_folder + "/" + model_file1
        model = model_loader.load_sklearn_model(model_file)

        train_percent = config["train_percent"]

        x_train, y_train = classifiers.split_dataset_train(
            x, y, train_percent)

        x_eval, y_eval = classifiers.split_dataset_test(
            x, y, train_percent)

        model, acc, diff, total, _ = classifiers.predict_decision_tree(
            model, x_train, y_train, False)
        diff_sum_train += diff
        total_sum_train += total

        model, acc, diff, total, _ = classifiers.predict_decision_tree(
            model, x_eval, y_eval, False)
        diff_sum_test += diff
        total_sum_test += total

        results_train.append([str(k + 1), str(j + 1), expname,
                              filename, str(diff_sum_train/total_sum_train*100)])

        results_test.append([str(k + 1), str(j + 1), expname,
                             filename, str(diff_sum_test/total_sum_test*100)])

with open(root_data_folder + "/output/" + "cross_check_" + output_file + "_train.csv", "w") as f:
    for r in results_train:
        f.write(",".join(r) + ",\n")

with open(root_data_folder + "/output/" + "cross_check_" + output_file + "_test.csv", "w") as f:
    for r in results_test:
        f.write(",".join(r) + ",\n")

import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import deep_learning
from modules import loader
from modules.preprocessing import Preprocessing
from modules import generator

import tensorflow as tf

from keras import backend as K

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

# root_data_folder += "/selected"
# filenames = ["exp_345", "exp_350", "exp_352", "exp_combined"]
use_rnn = True

use_top = True

if config["load_from_container"]:
    if use_rnn:
        # root_model_folder = config["root_model_container"] + "/deep_rnn_selected_all"
        root_model_folder = config["root_model_container"] + "/deep_rnn"
    else:
        root_model_folder = config["root_model_container"] + "/deep"

def list_files(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(
        join(mypath, f)) and '.h5' in f and ((use_top and "top" in f) or not use_top) and not ".txt" in f]
    return onlyfiles


model_files = list_files(root_model_folder)
print(model_files)

results_train = []
results_test = []


prep = Preprocessing()

if config["one_hot_encoding"]:
    prep.create_encoder(prep.adapt_input(
        generator.generate_binary(config["n_valves"])))

for (i, model_file1) in enumerate(model_files):
    # load model
    model_file = root_model_folder + "/" + model_file1
    for (j, filename) in enumerate(filenames):
        # load dataset
        data_file = root_data_folder + "/" + filename + ".csv"
        x, y, _, _ = loader.load_dataset(data_file)
        y = loader.binarize(y)

        if config["one_hot_encoding"]:
            y = prep.encode(prep.adapt_input(y))

        sizex = np.shape(x)

        graph = tf.Graph()
        with tf.Session(graph=graph):
            model = deep_learning.dl_load_model(model_file)

            train_percent = config["train_percent"]
            train_percent = 0

            x_train, y_train = classifiers.split_dataset_train(
                x, y, train_percent)

            x_eval, y_eval = classifiers.split_dataset_test(
                x, y, train_percent)

            # acc = deep_learning.eval_model(model, x_train, y_train, sizex[1])

            # results_train.append(
            #     [str(i + 1), str(j + 1), model_file, filename, str(acc*100)])

            acc = deep_learning.eval_model(
                model, x_eval, y_eval, sizex[1], use_rnn)

            results_test.append(
                [str(i + 1), str(j + 1), model_file, filename, str(acc*100)])

            print("model: " + model_file + "\tdataset: " +
                  data_file + "\tacc (test):" + str(acc))

# with open(root_data_folder + "/output/" + "cross_check_deep_train.csv", "w") as f:
#     for r in results_train:
#         f.write(",".join(r) + ",\n")
if use_rnn:
    output_file = "cross_check_deep_2_rnn_test.csv"
else:
    output_file = "cross_check_deep_1_test.csv"

with open(root_data_folder + "/output/" + output_file, "w") as f:
    for r in results_test:
        f.write(",".join(r) + ",\n")

import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import deep_learning, loader
import tensorflow as tf
from keras import backend as K
import time
import os
from shutil import copyfile, copy2
import yaml

from modules.preprocessing import Preprocessing
from modules import generator

prep = Preprocessing()

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]

# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]
model_filenames = filenames

custom_output = False

# custom_output = True
# root_data_folder += "/random1"
# # root_model_folder = "./data/models/deep_rnn_random"
# filenames = ["exp_179"]
# model_filenames = ["exp_39"]


# root_data_folder += "/control/2"
# filenames = ["exp_217"]
# model_filenames = ["exp_217"]


# set this as in saved models folder
n_reps = 5

results_vect_train = []
results_vect_test = []

use_rnn = True

output_filename = "eval_deep_1_"
if use_rnn:
    output_filename = "eval_deep_2_rnn_"

if custom_output:
    output_filename += "custom_"

# output_filename = "eval_deep_3_rnn_random_"
# output_filename = "eval_deep_5_rnn_random_"

# output_filename = "eval_deep"

if config["one_hot_encoding"]:
    prep.create_encoder(prep.adapt_input(
        generator.generate_binary(config["n_valves"])))

if config["load_from_container"]:
    if use_rnn:
        root_model_folder = config["root_model_container"] + "/deep_rnn_control"
    else:
        root_model_folder = config["root_model_container"] + "/deep"

# create separate models for each data file
for fn, filename in enumerate(filenames):
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, _, _ = loader.load_dataset(data_file)

    # process input data
    print("select data")

    # binarize the outputs
    y = loader.binarize(y)

    if config["one_hot_encoding"]:
        y = prep.encode(prep.adapt_input(y))

    train_percent = config["train_percent"]

    x_train, y_train = classifiers.split_dataset_train(
        x, y, train_percent)

    x_eval, y_eval = classifiers.split_dataset_test(
        x, y, train_percent)

    # bookmark_index = 1
    # x_train = x[0:bookmarks[bookmark_index], :]
    # y_train = y[0:bookmarks[bookmark_index], :]
    # x_eval = x[0:bookmarks[len(bookmarks)-1], :]
    # y_eval = y[0:bookmarks[len(bookmarks)-1], :]

    # x = loader.remove_col(x, 1)
    # y = loader.remove_col(y, 1)
    print("end select data")
    # quit(0)
    ## 

    sizex = np.shape(x_train)

    top_train = 0
    average_train = 0
    top_test = 0
    average_test = 0

    top_train_dt = 0
    average_train_dt = 0
    top_test_dt = 0
    average_test_dt = 0
    top_fsize = 0
    average_fsize = 0

    # run multiple evaluations (each training may return different results in terms of accuracy)
    for i in range(n_reps):
        print("evaluating model rep: " + str(i) + "/" + str(n_reps))

        # session = K.get_session()
        model_file = root_model_folder + "/" + model_filenames[fn]
        model_file_raw = model_file
        model_file_raw += "_" + str(i+1)
        model_file = model_file_raw + ".h5"

        # create tensorflow graph session
        graph = tf.Graph()
        with tf.Session(graph=graph):

            model = deep_learning.dl_load_model(model_file)

            acc_train = deep_learning.eval_model(
                model, x_train, y_train, sizex[1], use_rnn)

            dt = time.time()
            acc_test = deep_learning.eval_model(
                model, x_eval, y_eval, sizex[1], use_rnn)
            dt = time.time() - dt

            if acc_train > top_train:
                top_train = acc_train

            if acc_test > top_test:
                top_test = acc_test

            if dt > top_test_dt:
                top_test_dt = dt

            _, dt_train, fsize = loader.load_model_result_file(
                model_file + ".txt")

            if dt_train > top_train_dt:
                top_train_dt = dt_train

            if fsize > top_fsize:
                top_fsize = fsize

            average_test += acc_test
            average_train += acc_train
            average_test_dt += dt
            average_train_dt += dt_train
            average_fsize += fsize

    average_train = average_train/n_reps*100
    average_test = average_test/n_reps*100
    average_train_dt /= n_reps
    average_test_dt /= n_reps
    average_fsize /= n_reps

    results_vect_train.append(
        [filename, average_train, top_train * 100, average_train_dt, average_fsize])
    results_vect_test.append(
        [filename, average_test, top_test * 100, average_test_dt, average_fsize])

    # K.clear_session()


def extract_csv(vect):
    csvoutput = ""
    for row in vect:
        # print(e)
        for (i, e) in enumerate(row):
            if i == 0:
                csvoutput += e
            else:
                csvoutput += str(e)

            if i < len(row):
                csvoutput += ","

        csvoutput += "\n"

    return csvoutput


csvoutput = extract_csv(results_vect_train)

with open("./data/output/" + output_filename + "train.csv", "w") as f:
    f.write(csvoutput)

csvoutput = extract_csv(results_vect_test)

with open("./data/output/" + output_filename + "test.csv", "w") as f:
    f.write(csvoutput)

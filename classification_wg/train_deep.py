import numpy as np
import pandas as pd
import json

# import our modules
from modules import classifiers
from modules import deep_learning, loader
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import time
import os
from shutil import copyfile, copy2
import yaml
from modules.preprocessing import Preprocessing
from modules import generator
from modules import preprocessing
from sklearn.preprocessing import MinMaxScaler

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]
filename = config["filename"]
modelname = config["modelname"]

if config["run_clean"]:
    loader.clean(root_model_folder)

n_reps = 5
append_timestamp = False
save_best_model = True

if n_reps > 1:
    append_timestamp = False
    save_best_model = True
else:
    save_best_model = False

use_rnn = True
use_scaler = config["use_normalization"]

prep = Preprocessing()

data_file = root_data_folder + "/" + filename + ".csv"

df = loader.load_dataset_pd(data_file)
filter_labels = []
filter_labels = config["filter_labels"]

if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]

# df["duration"] > 0
df = df[df["volume"] >= 1]
# df = df[df["duration"] < 10]

df = loader.format_data(df, config["map_labels"])
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

if use_scaler:
    y = y[X[:, 1]<1000]
    X = X[X[:, 1]<1000]
    train_scaler = MinMaxScaler()
    train_scaler.fit(X)
    X = train_scaler.transform(X)

# convert categories to required format for neural networks (one-hot encoding)
y = to_categorical(y)
print(X)
print(y)

train_percent = config["train_percent"]

x_train, y_train = classifiers.split_dataset_train(
    X, y, train_percent)

x_eval, y_eval = classifiers.split_dataset_test(
    X, y, train_percent)

sizex = np.shape(x_train)

top_acc = 0
top_model_filename = None

print("x train")
print(x_train)
print("y train")
print(y_train)
print(np.shape(y_train))


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

# run multiple evaluations (each training may return different results in terms of accuracy)
for i in range(n_reps):
    print("evaluating model rep: " + str(i) + "/" + str(n_reps))
    # session = K.get_session()
    model_file = root_model_folder + "/" + modelname
    if use_rnn:
        model_file += "_rnn"
    else:
        model_file += "_dense"
    model_file_raw = model_file
    model_file_raw += "_" + str(i+1)
    if append_timestamp:
        app = "_" + str(time.time())
        model_file_raw += app
    model_file = model_file_raw + ".h5"

    # create tensorflow graph session
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph):
        acc_train = 0
        acc_test = 0
        if use_rnn:
            tstart = time.time()
            model = deep_learning.create_model_RNN(x_train, y_train, config["activation_fn"], config["loss_fn"])
            dt = time.time() - tstart
            deep_learning.dl_save_model(model, model_file)
            fsize = os.stat(model_file).st_size
            deep_learning.eval_write_info_RNN(
                model, x_eval, y_eval, model_file, dt, fsize)
            acc_train = deep_learning.eval_model_RNN(
                model, x_train, y_train, sizex[1])
            acc_test = deep_learning.eval_model_RNN(
                model, x_eval, y_eval, sizex[1])
        else:
            tstart = time.time()
            model = deep_learning.create_model(x_train, y_train, config["activation_fn"], config["loss_fn"])
            dt = time.time() - tstart
            deep_learning.dl_save_model(model, model_file)
            fsize = os.stat(model_file).st_size
            deep_learning.eval_write_info(
                model, x_eval, y_eval, model_file, dt, fsize)
            acc_train = deep_learning.eval_model(
                model, x_train, y_train, sizex[1], use_rnn)
            acc_test = deep_learning.eval_model(
                model, x_eval, y_eval, sizex[1], use_rnn)

        acc_train_vect["files"].append(model_file)
        acc_train_vect["acc"].append(acc_train)

        if acc_test > top_acc:
            top_acc = acc_test
            top_model_filename = model_file_raw

        acc_test_vect["files"].append(model_file)
        acc_test_vect["acc"].append(acc_test)

        if i == n_reps - 1:
            if save_best_model:
                copy2(top_model_filename + ".h5",
                        top_model_filename + "_top.h5")
                copy2(top_model_filename + ".h5.txt",
                        top_model_filename + "_top.h5.txt")

def set_avg(dc):
    # for each input file (experiment)
    acc_vect = np.array(dc["acc"])
    dc["avg"] = np.mean(acc_vect)
    dc["top"] = np.max(acc_vect)
    dc["best_model"] = dc["files"][np.argmax(acc_vect)]
    print(dc)

set_avg(acc_train_vect)
set_avg(acc_test_vect)

# print("\ntrain vect")
# print(json.dumps(acc_train_vect))
# print("\ntest vect")
# print(json.dumps(acc_test_vect))

def extract_csv(vect):
    csvoutput = ""
    csvoutput = str(vect["avg"]) + "," + str(vect["top"]) + \
        "," + str(vect["best_model"]) + "\n"
    return csvoutput

csvoutput = extract_csv(acc_train_vect)

if use_rnn:
    method = "rnn"
else:
    method = "dense"

with open("./output/eval_" + method + "_train.csv", "w") as f:
    f.write(csvoutput)

csvoutput = extract_csv(acc_test_vect)

with open("./output/eval_" + method + "_test.csv", "w") as f:
    f.write(csvoutput)
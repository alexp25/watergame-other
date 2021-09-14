import numpy as np
import pandas as pd

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

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]
filenames = config["filenames"]
filename = filenames[0]

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

use_rnn = False

prep = Preprocessing()

data_file = root_data_folder + "/" + filename + ".csv"
X, y, features, classes = loader.load_dataset(data_file, True)

print(y)

# convert categories to required format for neural networks (one-hot encoding)
y = to_categorical(y)

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

# run multiple evaluations (each training may return different results in terms of accuracy)
for i in range(n_reps):
    print("evaluating model rep: " + str(i) + "/" + str(n_reps))
    # session = K.get_session()
    model_file = root_model_folder + "/" + filename
    model_file_raw = model_file
    model_file_raw += "_" + str(i+1)
    if append_timestamp:
        app = "_" + str(time.time())
        model_file_raw += app
    model_file = model_file_raw + ".h5"

    # create tensorflow graph session
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph):
        if use_rnn:
            tstart = time.time()
            model = deep_learning.create_model_RNN(x_train, y_train, config["activation_fn"], config["loss_fn"])
            dt = time.time() - tstart
            deep_learning.dl_save_model(model, model_file)
            fsize = os.stat(model_file).st_size
            deep_learning.eval_write_info_RNN(
                model, x_eval, y_eval, model_file, dt, fsize)
            acc = deep_learning.eval_model_RNN(
                model, x_eval, y_eval, sizex[1])
        else:
            tstart = time.time()
            model = deep_learning.create_model(x_train, y_train, config["activation_fn"], config["loss_fn"])
            dt = time.time() - tstart
            deep_learning.dl_save_model(model, model_file)
            fsize = os.stat(model_file).st_size
            deep_learning.eval_write_info(
                model, x_eval, y_eval, model_file, dt, fsize)
            acc = deep_learning.eval_model(
                model, x_eval, y_eval, sizex[1], use_rnn)

        if acc > top_acc:
            top_acc = acc
            top_model_filename = model_file_raw

        if i == n_reps - 1:
            if save_best_model:
                copy2(top_model_filename + ".h5",
                        top_model_filename + "_top.h5")
                copy2(top_model_filename + ".h5.txt",
                        top_model_filename + "_top.h5.txt")


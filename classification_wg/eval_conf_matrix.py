import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import json

# import our modules
from modules import classifiers, model_loader
from modules import deep_learning, loader, graph
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
import time
from shutil import copyfile, copy2
import yaml
from modules.preprocessing import Preprocessing
from modules import generator
from modules import preprocessing
from sklearn.preprocessing import MinMaxScaler

def plot_confusion_matrix(l, p, c):
  # Confusion matrix
  conf_mat=confusion_matrix(l, p)
  # print(conf_mat)
  nb_classes = len(c)
  fig = plt.figure(figsize=(10,7))

  class_names = list(c)
  conf_mat = conf_mat.astype(float)

  for x in range(conf_mat.shape[0]):
    conf_mat[x] = conf_mat[x] / conf_mat[x].sum()

  df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names).astype(float)
  heatmap = sn.heatmap(df_cm, annot=True, cmap="mako")
  # "YlGnBu"

  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)

  plt.title('Confusion Matrix', fontsize=18)
  plt.ylabel('True label', fontsize=16)
  plt.xlabel('Predicted label', fontsize=16)
  plt.show()
  graph.save_figure(fig, "./figs/eval_conf_matrix")

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_model_folder = config["root_model_folder"]
filename = config["filename"]
modelname = config["modelname"]
model_to_eval = config["model_to_eval"]
use_normalization = config["use_normalization"]
data_file = root_data_folder + "/" + filename + ".csv"

df = loader.load_dataset_pd(data_file)
filter_labels = []
filter_labels = config["filter_labels"]
print(filter_labels)

if len(filter_labels) > 0:
    boolean_series = df['label'].isin(filter_labels)
    df = df[boolean_series]

df = df[df["volume"] >= 1]

df = loader.format_data(df, config["map_labels"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y, features, classes = loader.get_dataset_xy(df)
X = X.to_numpy()
y = y.to_numpy()

if use_normalization:
    y = y[X[:, 1]<1000]
    X = X[X[:, 1]<1000]
    train_scaler = MinMaxScaler()
    train_scaler.fit(X)
    X = train_scaler.transform(X)

y = to_categorical(y)

train_percent = config["train_percent"]

x_train, y_train = classifiers.split_dataset_train(
    X, y, train_percent)

x_eval, y_eval = classifiers.split_dataset_test(
    X, y, train_percent)

print(classes)
classes = config["class_labels"]
y_class_target = np.argmax(y_train, axis=1)
print('TARGET')
for i in range(len(classes)):
    print('Class = ', i)
    print(y_class_target[y_class_target == i].shape)


if "h5" in model_to_eval:
    model = deep_learning.dl_load_model(model_to_eval)
    y_pred = model.predict(x_train)
    print(y_pred.shape)
    y_class = np.argmax(y_pred, axis=1)
else:
    model = model_loader.load_sklearn_model(model_to_eval)
    # print_tree.plot_decision_tree(model, features, classes, "dtree.png")
    res = deep_learning.reshape_RNN(x_train)
    x_train = res[0]
    y_pred = model.predict(x_train)
    print(y_pred.shape)
    y_class = y_pred

print('PREDICTIONS')
for i in range(len(classes)):
    print('Class = ', i)
    print(y_class[y_class == i].shape)

print(np.sum(y_class == y_class_target) / y_class.shape[0])

plot_confusion_matrix(y_class_target, y_class, classes)

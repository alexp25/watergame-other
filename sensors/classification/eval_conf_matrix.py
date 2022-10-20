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
import apply_filters


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
# https://github.com/DTrimarchi10/confusion_matrix
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    fig = plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    
    return fig
    

def plot_confusion_matrix(l, p, c):
    # Confusion matrix
    conf_mat = confusion_matrix(l, p)
    # print(conf_mat)
    nb_classes = len(c)
    fig = plt.figure(figsize=(10, 7))

    class_names = list(c)
    conf_mat = conf_mat.astype(float)

    for x in range(conf_mat.shape[0]):
        conf_mat[x] = conf_mat[x] / conf_mat[x].sum()

    df_cm = pd.DataFrame(conf_mat, index=class_names,
                         columns=class_names).astype(float)
                         
    heatmap = sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    # "YlGnBu", "mako"

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=18)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=18)

    plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.show()
    graph.save_figure(fig, "./figs/eval_conf_matrix")


def plot_confusion_matrix_v2(y_class_target, y_class, classes):
    n_class = len(classes)
    categories = classes

    labels = ["" for c in range(n_class*n_class)]

    conf_mat = confusion_matrix(y_class_target, y_class)

    fig = make_confusion_matrix(conf_mat, 
                        group_names=labels,
                        categories=categories, 
                        cmap='binary')

    plt.show()
    graph.save_figure(fig, "./figs/eval_conf_matrix")
    return fig

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

df = apply_filters.apply_filter_labels(df, filter_labels)

if config["apply_balancing"]:
    df = apply_filters.apply_balancing(df, filter_labels)

df = loader.format_data(df, config["map_labels"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y, features, classes = loader.get_dataset_xy(df)
X = X.to_numpy()
y = y.to_numpy()

if use_normalization:
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
    if 'rnn' in model_to_eval:
        # res = deep_learning.reshape_RNN(x_train)
        # x_train = res[0]
        print(np.shape(x_train))
    y_pred = model.predict(x_train)
    print(y_pred.shape)
    y_class = np.argmax(y_pred, axis=1)
else:
    model = model_loader.load_sklearn_model(model_to_eval)
    # print_tree.plot_decision_tree(model, features, classes, "dtree.png")
    # res = deep_learning.reshape_RNN(x_train)
    # x_train = res[0]
    y_pred = model.predict(x_train)
    print(y_pred.shape)
    y_class = y_pred

print('PREDICTIONS')
for i in range(len(classes)):
    print('Class = ', i)
    print(y_class[y_class == i].shape)

print(np.sum(y_class == y_class_target) / y_class.shape[0])

plot_confusion_matrix(y_class_target, y_class, classes)
# plot_confusion_matrix_v2(y_class_target, y_class, classes)


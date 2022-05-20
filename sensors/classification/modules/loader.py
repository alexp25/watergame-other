import numpy as np
import csv
import os
import shutil
import pandas as pd


def clean(folder):
    except_files = [".gitkeep"]
    for filename in os.listdir(folder):
        if filename in except_files:
            continue

        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                print("removing file: " + file_path)
                os.unlink(file_path)

            elif os.path.isdir(file_path):
                print("removing folder: " + file_path)
                shutil.rmtree(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def load_model_result_file(filename):
    accuracy = 0
    dt = 0
    fsize = 0
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            if "accuracy" in line:
                accuracy = float(line.split("accuracy: ")[1])*100
            if "dt" in line:
                dt = float(line.split("dt: ")[1])
            if "fsize" in line:
                fsize = float(line.split("fsize: ")[1])
    return accuracy, dt, fsize


def format_data(df, map):
    # format the data, map classes to numbers
    df['label'] = df['label'].map(map)
    return df


def get_dataset_xy(df):
    # separate the feature (input) columns from the target column
    df = df[['uid', 'label', 'duration', 'volume']]
    df = df.fillna(0)
    target_col = 1
    features = list(df.columns)
    n_features = len(features)
    target_features = features[target_col]
    features = features[target_col+1:n_features]

    print("features: ")
    print(features)
    print("target: ")
    print(target_features)
    print("data: ")
    print(df)

    # get the last (target) column
    target = df.iloc[:, target_col]
    X = df[features]
    y = target

    # get unique values for classes
    classes = set(y)
    classes = list(classes)
    classes = [str(c) for c in classes]

    print(classes)

    return X, y, features, classes


def load_dataset_pd(input_file):
    # read the csv data (HR dataset)
    dataFrame = pd.read_csv(input_file, header=0)
    return dataFrame


def load_dataset(input_file):
    dataFrame = pd.read_csv(input_file, header=0)
    
    # get input/target datasets
    X, y, features, classes = get_dataset_xy(dataFrame)

    X = X.to_numpy()
    y = y.to_numpy()

    return X, y, features, classes


def load_dataset_old(filename):
    # load the dataset
    with open(filename, "r") as f:
        header = f.read().split("\n")[0].split(",")

    dataset = np.genfromtxt(filename, delimiter=',')
    # split into input (X) and output (y) variables
    x = dataset[1:, 2:13]
    y = dataset[1:, 14:20]

    sizex = np.shape(x)
    sizey = np.shape(y)

    print(sizex)
    print(sizey)

    print(x)
    print(y)

    return x, y, header[2:13], header[14:20]

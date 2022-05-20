import numpy as np
import csv
import os
import shutil
from datetime import datetime
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


def load_dataset_raw_buffer(filename):
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

    return x, y


def load_dataset(filename):
    # load the dataset
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        header = lines[0].split(",")

    x = np.genfromtxt(filename, delimiter=',')
    # x = np.nan_to_num(x)
    sizex = np.shape(x)
    print(sizex)
    # quit()
    return x, header


def format_data(df):
    # format the data, map classes to numbers
    d = {'chiuveta_rece': 1, 'chiuveta_calda': 2, "toaleta": 3, "dus": 4, "masina_spalat": 5, "masina_spalat_vase": 6}
    df['label'] = df['label'].map(d)
    return df

def load_dataset_pd(input_file, format=False):
    # read the csv data (HR dataset)
    dataFrame = pd.read_csv(input_file, header=0)
    if format:
        dataFrame = format_data(dataFrame)
    return dataFrame


def remove_col(x, col):
    # numpy.delete(arr, obj, axis=None)
    return np.delete(x, col, 1)


def remove_row(x, row):
    return np.delete(x, row, 0)


def binarize(x):
    s = np.shape(x)

    rows = s[0]
    cols = s[1]

    for i in range(0, rows):
        for j in range(0, cols):
            if x[i, j] > 0:
                x[i, j] = 1
            else:
                x[i, j] = 0

    return x


def save_as_csv_2d(filename, np2d):
    data = ""
    s = np.shape(np2d)
    for i in range(s[0]):
        for j in range(s[1]):
            data += str(np2d[i][j])+","
        data += "\n"
    with open(filename, "w") as f:
        f.write(data)

def save_as_csv_1d(filename, np1d):
    data = ""
    for el in np1d:
        data += str(el)+","
    with open(filename, "w") as f:
        f.write(data)

if __name__ == "__main__":
    clean("./data/models/crt")

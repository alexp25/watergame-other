import numpy as np
import pandas as pd
import time
import os
import yaml
from typing import List
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from modules import clustering
from modules import loader, graph
from modules.graph import Timeseries

def remove_outliers(data):
    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]
    for j in range(cols):
        for i in range(rows):
            if i > 1:
                if data[i][j] > 1200:
                    data[i][j] = data[i-1][j]
    return data


def reorder(x, order):
    x_ord = []
    for (i, ord) in enumerate(order):
        x_ord.append(x[ord])

    return np.array(x_ord)


def reorder2d(x, order):
    sdata = np.shape(x)
    rows = sdata[0]
    cols = sdata[1]

    x_ord = []
    for i in range(rows):
        new_row = []
        for (j, ord) in enumerate(order):
            new_row.append(x[i, ord])
        x_ord.append(new_row)

    return np.array(x_ord)


def create_timeseries(data, header, datax=None):
    tss: List[Timeseries] = []
    # colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']

    ck = 0
    sdata = np.shape(data)
    print(sdata)
    
    rows = sdata[0]
    cols = sdata[1]   

    colors = cm.rainbow(np.linspace(0, 1, cols))
    # colors = cm.viridis(np.linspace(0, 1, cols))

    for j in range(cols):
        ts: Timeseries = Timeseries()
        try:
            ts.label = header[j]
        except:
            ts.label = "unknown"
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        for i in range(rows):
            if datax is not None:
                ts.x.append(datax[i][j])
            else:
                ts.x.append(i)
            ts.y.append(data[i][j])
        tss.append(ts)
        ts = None

    return tss

def create_timeseries_rows(data, header, datax=None):
    tss: List[Timeseries] = []
    # colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']

    ck = 0
    sdata = np.shape(data)
    print(sdata)
    
    rows = sdata[0]
    cols = sdata[1]   

    colors = cm.rainbow(np.linspace(0, 1, cols))
    # colors = cm.viridis(np.linspace(0, 1, cols))

    for j in range(cols):
        ts: Timeseries = Timeseries()
        try:
            ts.label = header[j]
        except:
            ts.label = "unknown"
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        for i in range(rows):
            if datax is not None:
                ts.x.append(datax[i][j])
            else:
                ts.x.append(i)
            ts.y.append(data[i][j])
        tss.append(ts)
        ts = None

    return tss

def normalize_sum_axis(a, ax=0):
    new_matrix = normalize_sum_1(a, ax)
    # new_matrix = normalize1(new_matrix, 1-ax)
    return new_matrix

def normalize_sum_1(a, ax=0):
    row_sums = a.sum(axis=ax)
    if ax == 1:
        new_matrix = a / row_sums[:, np.newaxis]
    else:
        new_matrix = a / row_sums[np.newaxis, :]
    new_matrix = np.nan_to_num(new_matrix) 
    return new_matrix

def normalize_axis_01(a, ax=0):
    sizea = np.shape(a)
    min_vals = a.min(axis=ax)
    max_vals = a.max(axis=ax)
    new_matrix = a  
    for i in range(sizea[1-ax]):
        for j in range(sizea[ax]):
            new_matrix[i][j] = (a[i][j] - min_vals[i]) / (max_vals[i] - min_vals[i])
    new_matrix = np.nan_to_num(new_matrix) 
    return new_matrix
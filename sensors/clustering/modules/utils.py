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


def create_timeseries(data, header, datax=None, datax2=None):
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
            elif datax2 is not None:
                ts.x.append(datax2[i])    
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


def get_levels(stdev_min, stdev_max, n_clusters):
    # stdev_min *= 1.1
    # stdev_max *= 1.1
    # compute levels for stdev reclustering
    levels = np.linspace(stdev_min, stdev_max, n_clusters)
    # print(stdev_min, stdev_max)
    print(levels)
    levels = levels[1:]
    # recompute levels as exactly mid-levels (half distance between each original level)
    # levels = [0.001, 0.01, 0.02, 0.05]
    levels = -np.sort(-levels)
    return levels

def assign_levels(stdev_clusters, stdev_coords, levels):
    # recluster by stdev level
    stdev_coords_by_stdev = {}
    for level_idx, level in enumerate(levels):
        stdev_coords_by_stdev[str(level_idx)] = []

    new_assignments = []
    for k in stdev_clusters:
        for i, elem in enumerate(stdev_clusters[k]):
            elem_adj = "0"
            for level_idx, level in enumerate(levels):
                if elem < level:
                    elem_adj = str(len(levels) - level_idx - 1)
            new_assignments.append(int(elem_adj))
            # print(elem_adj)
            if stdev_coords is not None:
                elem_coord = stdev_coords[k][i]
                stdev_coords_by_stdev[elem_adj].append(elem_coord)

    return stdev_coords_by_stdev, new_assignments

def assign_levels_by_zones(stdev_clusters, stdev_coords, levels_zones):
    # recluster by stdev level
    stdev_coords_by_stdev = {}
    keys = list(levels_zones)
    for level_idx, level in enumerate(levels_zones[keys[0]]):
        stdev_coords_by_stdev[str(level_idx)] = []

    new_assignments = []
    for k in stdev_clusters:
        # print(k)
        for i, elem in enumerate(stdev_clusters[k]):
            elem_adj = "0"
            for level_idx, level in enumerate(levels_zones[k]):
                if elem <= level:
                    elem_adj = str(len(levels_zones[k]) - level_idx - 1)
            new_assignments.append(int(elem_adj))
            # print(elem_adj)
            if stdev_coords is not None:
                elem_coord = stdev_coords[k][i]
                stdev_coords_by_stdev[elem_adj].append(elem_coord)
    # quit()
    return stdev_coords_by_stdev, new_assignments

def check_hist(new_assignments, levels):
    d = {}
    for elem in new_assignments:  # pass through all the characters in the string
        if d.get(elem):  # verify if the character exists in the dictionary
            d[elem] += 1  # if it exist add 1 to the value for that character
        else:  # if it doesn’t exist initialize a new key with the value of the character
            d[elem] = 1  # and initialize the value (which is the counter) to 1

    d_vect = []
    for level in range(len(levels)):
        try:
            d_vect.append(d[level])
        except:
            d_vect.append(-1)
    return d_vect

def assign_levels_1d(stdev_clusters, stdev_coords, levels):
    # recluster by stdev level
    stdev_coords_by_stdev = {}
    for level_idx, level in enumerate(levels):
        stdev_coords_by_stdev[str(level_idx)] = []

    new_assignments = []
    for i, k in enumerate(stdev_clusters):
        elem = stdev_clusters[k]
        elem_adj = "0"
        for level_idx, level in enumerate(levels):
            if elem < level:
                elem_adj = str(len(levels) - level_idx - 1)
        # print(elem)
        # print(int(elem_adj))
        new_assignments.append(int(elem_adj))
        if stdev_coords is not None:
            elem_coord = stdev_coords[i]
            stdev_coords_by_stdev[elem_adj].append(elem_coord)

    # print(new_assignments)

    # labels = [str(level) for level in range(len(levels))]
    # labels

    return stdev_coords_by_stdev

def get_split_d_vect(d_vect):
    split_d_vect = []
    for i, d in enumerate(d_vect):
        d_vect_new = []
        for i1, d1 in enumerate(d_vect):
            if i1 == i:
                d_vect_new.append(d)
            else:
                d_vect_new.append(0)
        split_d_vect.append(d_vect_new)
    return split_d_vect
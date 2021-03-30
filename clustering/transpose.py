import os
import pandas as pd
import numpy as np
from modules import utils, graph, clustering
import matplotlib.pyplot as plt
from scipy import interpolate
import csv

folder = './data/Smart Water Meter/processed'
df = pd.read_csv(folder + "/" + "avg_out_2.csv")
x = df.to_numpy()
x = np.transpose(x)

file = open(folder + "/" + "avg_out_2t.csv", 'w', newline='')

# writing the data into the file
with file:
    write = csv.writer(file)
    write.writerows(x)
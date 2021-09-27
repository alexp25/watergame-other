
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math


def normalize(data):
    # data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)
    return scaled

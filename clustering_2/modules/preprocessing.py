
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import math


def normalize(data):
    # data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)
    return scaled

def imputation(data):
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(data)
    # data = imp.transform(data)
    imp = SimpleImputer(missing_values=0, strategy='mean')
    imp.fit(data)
    data = imp.transform(data)
    return data

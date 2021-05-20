import pandas as pd
from sklearn.cluster import OPTICS, KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import rcParams


def get_trend(series, seasonal=24, robust=False):
    result = seasonal_decompose(series, period=24, extrapolate_trend='freq')
    return result.trend

if __name__ == "__main__":
    if len(sys.argv) < 2:
        fn = "data/Water weekly/water_avg_weekly.csv"
        samples = 100
        n_clusters = 4
    else:
        fn = sys.argv[1]
        samples = int(sys.argv[2])
        n_clusters = int(sys.argv[3])
    df = pd.read_csv(fn, sep=',', header=None)
    s = np.shape(df)
    print(s)
    startindex = 3
    X = np.array(df.iloc[:, startindex:])

    for i in range(s[0]):
        trend = get_trend(X[i, :], 24, False)
        # print(trend)
        # quit()
        X[i, :] = np.nan_to_num(trend)
        df.iloc[i, startindex:] = X[i, :]

    df.to_csv("data/Water weekly/water_avg_weekly_trends.csv", header=False, index=False)

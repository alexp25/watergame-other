import os
import pandas as pd
import numpy as np
from modules import utils, graph, clustering
import matplotlib.pyplot as plt
from scipy import interpolate
import csv

folder = './data/Smart Water Meter/extracted'
folder_out = './data/Smart Water Meter/processed'

# folder = './data/Shower water/extracted'
# folder_out = './data/Shower water/processed'


def extract_timestamps(df, parse):
    tstamp = df["time"]
    # dates = tstamp.dt.strftime('%Y-%m-%d %H:%M')
    if parse:
        dates = tstamp
        date0 = np.min(tstamp)
        print(date0)
        # pd.Timestamp("1970-01-01")
        tstamp = [[((ts - date0) // pd.Timedelta('1h'))
                   for ts in dates]]
    tstamp = np.array(tstamp)
    tstamp = np.transpose(tstamp)
    return tstamp


xavg_vect = []

ndiv = 168


with os.scandir(folder) as entries:
    for index, entry in enumerate(entries):
        print(entry.name)

        try:
            df = pd.read_csv(folder + "/" + entry.name, parse_dates=['time'])
        except Exception as e:
            print(e)
            continue

        df['timex'] = df['time'].apply(lambda x: (
            x - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))
        df = df.sort_values(by=["timex"])

        print(df["time"])

        # print(weeks)
        tstamp = extract_timestamps(df, True)

        print(np.shape(tstamp))

        # quit()

        x = [df["data"]]
        x = np.transpose(x)
        tss = utils.create_timeseries(x, None, tstamp)

        # plt.plot(tstamp, x)
        # plt.show()

        # fig = graph.plot_timeseries_multi_sub2(
        #     [tss], ["test"], "x", ["y"], None, None, None)

        try:
            # groupby your key and freq
            g = df.groupby(pd.Grouper(key='time', freq='W'))
            # B: business day frequency
            # C: custom business day frequency
            # D: calendar day frequency
            # W: weekly frequency
            # M: month end frequency
        except Exception as e:
            print(e)
            continue

        # groups to a list of dataframes with list comprehension
        dfs = [group for _, group in g]

        print("nweeks: ", len(dfs))

        xavg = None
        xall = []

        for df_index, df in enumerate(dfs):
            x = df["data"]
            # exclude nan data
            array_sum = np.sum(x)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                continue
            x = [x]
            tstamp = extract_timestamps(df, True)
            tstamp_list = np.transpose(tstamp).tolist()[0]
            ndiv_orig = len(tstamp_list)
            try:
                tmin = tstamp_list[0]
                tmax = tstamp_list[ndiv_orig - 1]

                # interpolate/handle missing data
                f = interpolate.interp1d(tstamp_list, list(
                    x[0]), fill_value="extrapolate")
                # f = interpolate.interp1d(tstamp_list, list(x[0]))

                # resample dataframe weekly
                tnew = np.linspace(tmin, tmax, num=ndiv)
                xnew = f(tnew)

                # exclude nan data
                array_sum = np.sum(xnew)
                array_has_nan = np.isnan(array_sum)
                if not array_has_nan:
                    xall.append(xnew)

                # xnew = [xnew]
                # xnew = np.transpose(xnew)
                # tss = utils.create_timeseries(xnew, None, None)
                # fig = graph.plot_timeseries_multi_sub2(
                #     [tss], ["test"], "x", ["y"], None, None, None)
            except Exception as e:
                print(e)

        xall = np.array(xall)        
        try:
            xall = clustering.remove_outliers(xall)
        except Exception as e:
            print(e)
        xavg = np.average(xall, axis=0)
        xavg_vect.append(xavg)
        # if index == 2:
        #     break

    # print(xavg_vect)
    xavg_vect = np.array(xavg_vect)
    xavg_vect_disp = np.transpose(xavg_vect)
    tss = utils.create_timeseries(xavg_vect_disp, None, None)
    fig = graph.plot_timeseries_multi_sub2(
        [tss], ["test"], "x", ["y"], None, None, None)

    head = []
    for n in range(ndiv):
        head.append(str(n))

    # opening the csv file in 'w+' mode
    file = open(folder_out + "/" + "avg.csv", 'w', newline='')

    # writing the data into the file
    with file:
        file.write(",".join(head) + "\n")
        write = csv.writer(file)
        write.writerows(xavg_vect)

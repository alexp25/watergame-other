

# import our modules
from fileinput import filename
from modules import loader
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser

import yaml
config = yaml.safe_load(open("config.yml"))

root_data_folder = "./data"
# read the data from the csv file

extract_locations = False

# load sensors list
data_file = root_data_folder + "/" + "setup.csv"
df = loader.load_dataset_pd(data_file)

sensor_list = []

combined_sink_data = config["combined_sink_data"]
extract_inst_flow = config["extract_inst_flow"]

for row in df.iterrows():
    rowspec = row[1]
    # if not np.isnan(rowspec["id"]):
    if len(rowspec["id"]) > 0:
        sensor_spec = {
            "id": int(rowspec["id"]),
            "loc": rowspec["apartament"] + " - " + rowspec["loc"],
            "user": rowspec["apartament"],
            "labels": []
        }
        data_labels = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
        for dl in data_labels:
            try:
                if np.isnan(rowspec[dl]):
                    pass
            except:
                label = rowspec[dl]
                if extract_locations:
                    if label in ['chiuveta_calda', 'chiuveta_rece']:
                        label = label + '_' + rowspec["tip_loc"]
                sensor_spec["labels"].append(label)
                pass
        sensor_list.append(sensor_spec)
        print(sensor_spec)


# create separate models for each data file
for plot_index, sensor_spec in enumerate(sensor_list):
    filename = "watergame_sample_consumer_" + str(sensor_spec["id"]) + ".csv"
    data_file = root_data_folder + "/" + filename
    try:
        x, header = loader.load_dataset(data_file)
    except:
        continue

    df = loader.load_dataset_pd(data_file)
    n_spec_cols = 3
    n_chan = len(df.columns) - n_spec_cols

    for i, label in enumerate(sensor_spec["labels"]):
        if extract_inst_flow:
            df[label] = df.iloc[:, n_spec_cols+i*2]
            df[label+"_"] = df.iloc[:, n_spec_cols+i*2+1]
        else:
            df[label] = df.iloc[:, n_spec_cols+i*2+1]
            df[label+"_"] = df.iloc[:, n_spec_cols+i*2]

    cols = [n_spec_cols+i for i in range(n_chan)]
    df.drop(df.columns[cols], axis=1, inplace=True)

    df.insert(loc=1, column='user', value=sensor_spec['user'])

    if combined_sink_data:
        if "chiuveta_calda" in sensor_spec["labels"]:
            combined = df["chiuveta_rece"] + df["chiuveta_calda"]
            df.insert(loc=n_spec_cols, column="chiuveta", value=combined)
            df.insert(loc=n_spec_cols+1, column="chiuveta_", value=combined)
            # df["chiuveta_"] = df["chiuveta"]
            df.drop(["chiuveta_calda"], axis=1, inplace=True)
            df.drop(["chiuveta_rece"], axis=1, inplace=True)
            df.drop(["chiuveta_calda_"], axis=1, inplace=True)
            df.drop(["chiuveta_rece_"], axis=1, inplace=True)

    # print(sensor_spec)
    # print(df)
    # print(np.unique(df['user']))
    # quit()

    timestamps = df["timestamp"]
    timestamp_ref = "2021-10-31 04:00:00"

    format_ts = "%Y-%m-%d %H:%M:%S"

    df["timestamp"] = pd.to_datetime(df["timestamp"], format=format_ts)
    df["timestamp"] = df["timestamp"].apply(lambda x: x.timestamp())

    print(df["timestamp"])

    # quit()

    ts_ref = datetime.strptime(timestamp_ref, format_ts).timestamp()

    print(ts_ref)
    print(df["timestamp"][0])

    df['timestamp'] = np.where(
        (df['timestamp'] - ts_ref) < 0, df['timestamp']-(1*60*60), df['timestamp'])
    print(df["timestamp"])

    # adjust for daylight saving
    df["timestamp"] = df['timestamp'].apply(
        lambda x: pd.to_datetime(x, unit='s'))
    print(df)

    # quit()
    filename = "watergame_sample_consumer_" + \
        str(sensor_spec["id"])
    if combined_sink_data:
        filename += "_combined"
    filename += "_adapted"
    filename += ".csv"
    data_file = root_data_folder + "/" + filename
    loader.write_dataset_pd(df, data_file)



# import our modules
from fileinput import filename
from modules import loader
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser

root_data_folder = "./data"
# read the data from the csv file

extract_locations = False

# load sensors list
data_file = root_data_folder + "/" + "setup.csv"
df = loader.load_dataset_pd(data_file)

sensor_list = []
sensor_list_exp = []

for row in df.iterrows():
    rowspec = row[1]
    if not np.isnan(rowspec["id"]):
        sensor_spec = {
            "id": int(rowspec["id"]),
            "loc": rowspec["apartament"] + " - " + rowspec["loc"],
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

        for k, label in enumerate(sensor_spec["labels"]):
            sensor_list_exp.append({
                "online": False,
                "id": int(rowspec["id"]),
                "uid": str(sensor_spec["id"]) + "_" + str(k + 1),
                "label": label,
                "data": [],
                "ts": []
            })
        # print(row)
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

 
    # df["timestamp"] = df["timestamp"].strftime(format_ts)
    df["timestamp"] = df['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s'))
    print(df)
    
    # quit()
    filename = "watergame_sample_consumer_" + str(sensor_spec["id"]) + "_adapted.csv"
    data_file = root_data_folder + "/" + filename
    loader.write_dataset_pd(df, data_file)

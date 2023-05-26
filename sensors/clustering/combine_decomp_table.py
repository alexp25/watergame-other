

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np

import yaml
config = yaml.safe_load(open("config.yml"))

root_data_folder = "./data"
# read the data from the csv file

extract_locations = False
combined_sink_data = False

# load sensors list
data_file = root_data_folder + "/" + "setup.csv"
df = loader.load_dataset_pd(data_file)

sensor_list = []
sensor_list_exp = []

for row in df.iterrows():
    rowspec = row[1]
    # if not np.isnan(rowspec["id"]):
    if len(rowspec["id"]) > 0:
        sensor_spec = {
            "id": int(rowspec["id"]),
            "loc": rowspec["apartament"] + " - " + rowspec["loc"],
            'user': rowspec["apartament"],
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

        if combined_sink_data:
            labels = sensor_spec["labels"]
            sensor_spec["labels"] = []
            for label in labels:
                if ("chiuveta_rece" in label) or ("chiuveta_calda" in label):
                    if not "chiuveta" in sensor_spec["labels"]:
                        sensor_spec["labels"].append("chiuveta")
                else:
                    sensor_spec["labels"].append(label)

        for k, label in enumerate(sensor_spec["labels"]):
            sensor_list_exp.append({
                "online": False,
                "id": int(rowspec["id"]),
                "uid": str(sensor_spec["id"]) + "_" + str(k + 1),
                "label": label,
                "user": sensor_spec["user"],
                "data": [],
                "ts": []
            })
        # print(row)
        sensor_list.append(sensor_spec)
        print(sensor_spec)

# quit()
extract_inst_flow = config["extract_inst_flow"]

start_index = 1
# end_index = 100
end_index = 20160  # 2 weeks in minutes
# end_index = 200
# end_index = None

start_col = 3
end_col = None
fill_start = False

data_row_max_size = 0

# filter_id = [41364]
filter_id = []

# create separate models for each data file
for plot_index, sensor_spec in enumerate(sensor_list):
    print(sensor_spec["id"])
    if (len(filter_id) == 0) or (sensor_spec["id"] in filter_id):
        print("MATCH")
        filename = "watergame_sample_consumer_" + str(sensor_spec["id"])
        if combined_sink_data:
            filename += "_combined_adapted"
        filename += ".csv"
        data_file = root_data_folder + "/" + filename
        try:
            x, header = loader.load_dataset(data_file)
        except:
            continue

        df = loader.load_dataset_pd(data_file)   
        timestamps = df["timestamp"]
        sid = df["sensorId"]

        nheader = len(header)
        sx = np.shape(x)

        if fill_start:
            x = x[start_index:, :]
            x[:, 0:start_col-1] = np.transpose(np.array([[0] * (sx[0]-1)]))
        else:
            x = x[start_index:, start_col:]

        if end_index is not None:
            x = x[:end_index, :]
        if end_col is not None:
            x = x[:, :end_col]

        xlabel = ""
        xtype = ""

        if extract_inst_flow:
            xlabel = "flow [L/h]"
            xtype = "flow"
            # title = "consumption data (flow) node " + str(sid[0]) + " (" + sensor_spec["loc"] + ")"
            # title = "consumption data (flow) node " + str(sid[0])
            title = "flow data node " + str(sid[0])
            x = x[:, 1::2]
        else:
            xlabel = "volume [L]"
            xtype = "volume"
            title = "consumption data (volume) node " + \
                str(sid[0]) + " (" + sensor_spec["loc"] + ")"
            x = x[:, 0::2] / 1000

        sx = np.shape(x)
        print(sx)

        print("start")

        header = []
        for d in range(nheader-1):
            header.append(str(d+1))

        header = sensor_spec["labels"]
        mapping = config["name_mapping"]    
        header = [mapping[head] for head in header]
        
        # time axis labels
        # xlabels = [str(i) for i in range(sx[1])]
        xlabels = [str(i) for i in range(len(timestamps))]
        xlabels = timestamps
        xlabels = [np.datetime64(ts) for ts in timestamps]
        # xlabels = np.array(xlabels)
        # xlabels = np.transpose(xlabels)
        # print(xlabels)

        # aggregate data
        for label_index, label in enumerate(sensor_spec["labels"]):
            for sensor_exp in sensor_list_exp:  
                if sensor_exp["id"] == sensor_spec["id"] and sensor_exp["label"] == label:
                    try:
                        sensor_exp["data"] = x[:, label_index]
                        sensor_exp["ts"] = xlabels
                        data_row_max_size_1 = len(sensor_exp["data"])
                        if data_row_max_size_1 > data_row_max_size:
                            data_row_max_size = data_row_max_size_1
                        sensor_exp["online"] = True
                    except:
                        print("exc " + str(sensor_exp["id"]))


def process_fn():
    exp_data = "timestamp,user,source,value\n"
    for sexp in sensor_list_exp:
        if sexp["online"]:            
            for index, data in enumerate(sexp['data']):
                exp_data += str(sexp['ts'][index]) + ',' + sexp["user"] + "," + sexp["label"] + ',' + str(data) + '\n'
    return exp_data


def save_result(name, exp_data):
    result_name = root_data_folder + "/" + name
    result_name += ".csv"

    with open(result_name, "w") as f:
        f.write(exp_data)


exp_data = process_fn()
save_result('decomp_table', exp_data)

# if save_file:
#     # print(sensor_list_exp)
#     # exp_data = "uid,lat,lng,"
#     result_name = "res" if extract_inst_flow else "res_vol"
#     exp_data = process_fn("data")
#     save_result(result_name, exp_data)
#     exp_data = process_fn("ts")
#     save_result(result_name + "_ts", exp_data)
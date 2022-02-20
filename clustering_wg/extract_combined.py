

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np

root_data_folder = "./data"
# read the data from the csv file


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
                sensor_spec["labels"].append(rowspec[dl])
                pass
        for k, label in enumerate(sensor_spec["labels"]):
            sensor_list_exp.append({
                "online": False,
                "id": int(rowspec["id"]),
                "uid": str(sensor_spec["id"]) + "_" + str(k + 1),
                "label": label,
                "data": []
            })
        # print(row)
        sensor_list.append(sensor_spec)
        print(sensor_spec)


plot_all_data = True
# plot_all_data = False

remove_outlier = True
remove_outlier = False

# norm = True
norm = False

extract_inst_flow = False
extract_inst_flow = True

save_file = True

rolling_filter = True
rolling_filter = False


start_index = 1
# end_index = 100
end_index = None
start_col = 3
end_col = None
fill_start = False

data_row_max_size = 0

# create separate models for each data file
for plot_index, sensor_spec in enumerate(sensor_list):
    filename = "watergame_sample_consumer_" + str(sensor_spec["id"]) + ".csv"
    data_file = root_data_folder + "/" + filename
    try:
        x, header = loader.load_dataset(data_file)
    except:
        continue

    df = loader.load_dataset_pd(data_file)
    # timestamps = df[["timestamp"]]
    timestamps = df["timestamp"]
    sid = df["sensorId"]
    # print(list(timestamps))
    # quit()

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

    if rolling_filter:
        kernel_size = int(0.1 * sx[0])
        kernel = np.ones(kernel_size) / kernel_size
        for dim in range(sx[1]):
            x[:, dim] = np.convolve(x[:, dim], kernel, mode='same')

    if remove_outlier:
        outlier = -1
        outliers = []
        x = clustering.remove_outliers(x)

    sx = np.shape(x)
    print(sx)

    print("start")

    header = []
    for d in range(nheader-1):
        header.append(str(d+1))
    header = sensor_spec["labels"]
    mapping = {
        "dus": "shower",
        "chiuveta_rece": "sink_cold",
        "chiuveta_calda": "sink_hot",
        "toaleta": "toilet",
        "masina_spalat": "washing_machine",
        "masina_spalat_vase": "dishwasher"
    }
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
                    data_row_max_size_1 = len(sensor_exp["data"])
                    if data_row_max_size_1 > data_row_max_size:
                        data_row_max_size = data_row_max_size_1

                    sensor_exp["online"] = True
                except:
                    print("exc " + str(sensor_exp["id"]))

    if plot_all_data:
        xplot = x
        tss = utils.create_timeseries(xplot, header, None)
        xlabels = None
        fig = graph.plot_timeseries_multi_sub2(
            [tss], [title], "sample", [xlabel], (8, 6), None, None, xlabels, False, plot_index)
        result_name = "./figs/consumer_data_" + xtype + "_"
        if rolling_filter:
            result_name += "rf_"
        result_name += str(sid[0])
        graph.save_figure(fig, result_name, 200)

    # quit()

print(data_row_max_size)
# quit()

if save_file:
    # print(sensor_list_exp)
    # exp_data = "uid,lat,lng,"
    exp_data = "uid,label,x"
    for d in range(data_row_max_size):
        exp_data += "," + str(d)
    exp_data += "\n"
    for sexp in sensor_list_exp:
        if sexp["online"]:
            exp_data_row = sexp["uid"] + "," + sexp["label"] + ",x"
            len_data = len(sexp["data"])
            for d in sexp["data"]:
                exp_data_row += "," + str(d)
            len_data_rem = data_row_max_size - len_data
            for d in range(len_data_rem):
                exp_data_row += ",0"
            exp_data_row += "\n"
            exp_data += exp_data_row

    result_name = root_data_folder + "/res"
    if rolling_filter:
        result_name += "_rf"
    result_name += ".csv"

    with open(result_name, "w") as f:
        f.write(exp_data)

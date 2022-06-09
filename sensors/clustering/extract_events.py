

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np
import yaml
config = yaml.safe_load(open("config.yml"))

extract_inst_flow = config["extract_inst_flow"]

root_data_folder = "./data"
# read the data from the csv file

res_name = "res"
if not extract_inst_flow:
    res_name = "res_vol"

result_name = root_data_folder + "/" + res_name
result_name += ".csv"

result_ts_name = root_data_folder + "/" + res_name + "_ts"
result_ts_name += ".csv"

start_col = 3

x, header = loader.load_dataset(result_name)
df = loader.load_dataset_pd(result_name)

t, _ = loader.load_dataset(result_ts_name)
df_ts = loader.load_dataset_pd(result_ts_name)

x = df.to_numpy()
print(x)

x_ts = df_ts.to_numpy()

print(x_ts)

nheader = len(header)

sx = np.shape(x)
print(sx)

print("start")

events = []

for i in range(sx[0]):
    print(i)
    data = x[i, start_col:]
    data_ts = x_ts[i, start_col:]
    print(data)

    batch = []
    batch_idx = []
    batch_start_ts = []
    current_batch = []
    current_batch_idx = []
    batch_add_started = False
    
    for k in range(np.size(data, axis=0)):
        if batch_add_started:
            if data[k] > 0:
                current_batch.append(data[k])
                current_batch_idx.append(k)
            else:
                batch_add_started = False
                batch.append(np.array(current_batch))
                batch_idx.append(current_batch_idx)
                batch_start_ts.append(data_ts[current_batch_idx[0]])
                current_batch = []
                current_batch_idx = []
        else:
            if data[k] > 0:
                current_batch.append(data[k])
                current_batch_idx.append(k)
                batch_add_started = True

    print(batch)
  
    # cond = np.logical_and(data > 0, data > 0)
    # batch = np.split(data[cond], np.where(
    #     np.diff(np.where(cond)[0]) > 1)[0] + 1)


    for j, event in enumerate(batch):
        # print(event)
        duration = np.size(event)
        volume = np.mean(event)
        volume = np.sum(event)
        new_event = {
            "sid": x[i, 0],
            "event": j,
            "label": x[i, 1],
            "duration": duration,
            "volume": volume,
            "ts": batch_start_ts[j]
        }
        # print(duration, average)
        print(new_event)
        # quit()
        events.append(new_event)

exp_data = "uid,label,no,duration,volume,timestamp\n"
for evt in events:
    exp_data_row = str(evt["sid"]) + "," + str(evt["label"]) + "," + str(
        evt["event"]) + "," + str(evt["duration"]) + "," + str(evt["volume"]) + "," + str(evt["ts"]) + "\n"
    exp_data += exp_data_row

result_name = root_data_folder + "/res_evt"
if not extract_inst_flow:
    result_name += "_vol"
result_name += ".csv"

with open(result_name, "w") as f:
    f.write(exp_data)

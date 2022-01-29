

# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np


root_data_folder = "./data"
# read the data from the csv file

result_name = root_data_folder + "/res"
result_name += ".csv"

plot_all_data = True
# plot_all_data = False

start_index = 1
# end_index = 100
end_index = None
start_col = 3
end_col = None
fill_start = False

x, header = loader.load_dataset(result_name)
df = loader.load_dataset_pd(result_name)

x = df.to_numpy()
print(x)

nheader = len(header)

sx = np.shape(x)
print(sx)

print("start")

events = []

for i in range(sx[0]):
    print(i)
    data = x[i, start_col:]
    print(data)
    cond = np.logical_and(data > 0, data > 0)
    batch = np.split(data[cond], np.where(
        np.diff(np.where(cond)[0]) > 1)[0] + 1)

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
            "volume": volume
        }
        # print(duration, average)
        print(new_event)
        # quit()
        events.append(new_event)

exp_data = "uid,label,no,duration,volume\n"
for evt in events:
    exp_data_row = str(evt["sid"]) + "," + str(evt["label"]) + "," + str(
        evt["event"]) + "," + str(evt["duration"]) + "," + str(evt["volume"]) + "\n"
    exp_data += exp_data_row

result_name = root_data_folder + "/res_evt"
result_name += ".csv"

with open(result_name, "w") as f:
    f.write(exp_data)

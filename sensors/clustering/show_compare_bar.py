


# import our modules
from fileinput import filename
from modules import loader, graph
from modules import clustering
from modules import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yaml
import traceback
config = yaml.safe_load(open("config.yml"))

root_data_folder = "./data"
# read the data from the csv file

rolling_filter = False
# rolling_filter = True

fname_dict = config["fname_dict"]
title_dict = config["title_dict"]

data_files = ["res_" + key for key in fname_dict.keys()]
names = [title_dict[key] for key in title_dict.keys()]

# plot_all_data = True
plot_all_data = False

rolling_filter = True
rolling_filter = False

start_index = 1
# end_index = 100
end_index = None
start_col = 1
end_col = None
fill_start = False

savefig = False

nc = 3

key = "disp"
# key = "count"
# key = "duration"
# key = "volume"

cluster_specs_dict = {}

for df in data_files:
    # df = loader.load_dataset_pd(root_data_folder + "/" + df)
    result_name = root_data_folder + "/" + df + ".csv"
    try:
        x, header = loader.load_dataset(result_name)
        x = x[start_index:, start_col:]
        labels = np.shape(x)[0]
        # print(labels)
        for lb in range(labels):
            if not lb in cluster_specs_dict:
                cluster_specs = {
                    "label": str(lb),
                    "count": [x[lb,0]],
                    "disp": [x[lb,1]],
                    "duration": [x[lb,2]],
                    "volume": [x[lb,3]]
                }
                cluster_specs_dict[lb] = cluster_specs
            else:
                cluster_specs_dict[lb]["count"].append(x[lb,0])
                cluster_specs_dict[lb]["disp"].append(x[lb,1])
                cluster_specs_dict[lb]["duration"].append(x[lb,2])
                cluster_specs_dict[lb]["volume"].append(x[lb,3])
    except:
        traceback.print_exc()

# print(cluster_specs_dict)
labels = list(cluster_specs_dict.keys())
print(labels)

# plt.bar(labels, [1,1,1])
# plt.xlabel("Cluster")
# plt.ylabel("Value")
# plt.title("Cluster distribution")
# plt.show()


cmap = matplotlib.cm.get_cmap('viridis')
color_scheme = [cmap(i) for i in np.linspace(0, 1, len(data_files))]
print(color_scheme)

# quit()

# [1,2,3,2],[4,5,6,1],[2,3,4,5],[5,4,3,2]
# first pass (left - right) = [1,2,3,2]
# second pass (left - right) = [4,5,6,1]
# etc

print(labels)

specs = []
# extract spec
for k in cluster_specs_dict.keys():
    specs.append(cluster_specs_dict[k][key])

specs_mat = np.array(specs)
specs_mat = np.transpose(specs_mat)
spect_mat = list(specs_mat)

m = np.mean(specs_mat, axis=1)

for i, name in enumerate(names):
    try:
        print(name + ": " + str(m[i]))
    except:
        pass
# quit()
print(m)

# print(np.transpose(specs_mat))
# quit()
fig, _ = graph.plot_barchart_multi_core_raw(specs_mat, color_scheme, names, "Cluster", "Dispersion",
                                          "Within-cluster dispersion", None, None, True, None, 0, None)
# fig, _ = graph.plot_barchart_multi_core_raw([[1,2,3],[4,5,6],[2,3,4],[5,4,3]], color_scheme, names, "Cluster", "Value",
#                                           "Cluster distribution", labels, None, True, None, 0, None)

result_name = "./figs/event_clusters_distribution_" + key + ".png"
fig.savefig(result_name, dpi=200)
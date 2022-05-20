

from modules import loader, graph, utils
import numpy as np

filenames = ["clusters_inertia.csv", "clusters_dist.csv",
             "clusters_trends_inertia.csv", "clusters_trends_dist.csv"]
d2 = []
for f in filenames:
    d = loader.load_dataset(f)[0]
    d = utils.normalize_axis_01(np.array([d]), 1).tolist()[0]
    # d = d.tolist()
    d2.append(d)

d2np = np.array(d2)
d2np = np.transpose(d2np)
s = np.shape(d2np)

xlabels = [(i+1) for i in list(range(s[0]))]

print(xlabels)
xlabels = None

print(d2np)

# xlabels = None
# quit()

tss = utils.create_timeseries_rows(d2np, ["A1", "A2", "B1", "B2"], None)
fig = graph.plot_timeseries_multi_sub2(
    [tss], ["Optimal number of clusters"], "Number of clusters", ["Distortion"], None, None, None, xlabels)

# fig = graph.plot(distortions, list(n_clusters), "Optimal number of clusters", "Number of clusters", "Distortion")
# print(d)

# fig = graph.plot(distortions, list(n_clusters), "Optimal number of clusters", "Number of clusters", "Distortion")
# graph.save_figure(fig, "./figs/eval_dist.png")
# print(distortions)
# loader.save_as_csv_1d("clusters_dist.csv", distortions)

# fig = graph.plot(WCSS_vect, list(n_clusters), "Optimal number of clusters", "Number of clusters", "WCSS")
# graph.save_figure(fig, "./figs/eval_inertia.png")
# print(WCSS_vect)
# loader.save_as_csv_1d("clusters_inertia.csv", WCSS_vect)
# quit()

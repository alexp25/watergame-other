

# import our modules
from modules import graph
import numpy as np
import matplotlib


labels = ["silhouette score", "rand index",
          "adjusted rand index", "purity", "entropy"]
legend = ["", "patterns raw", "patterns filtered",
          "trends raw", "trends filtered"]

# 1 - higher is better, 0 - lower is better
order = [1, 1, 1, 1, 0]

print(labels)
print(legend)

cmap = matplotlib.cm.get_cmap('viridis')
color_scheme = [cmap(i) for i in np.linspace(0, 1, len(labels))]
print(color_scheme)


specs = [[0.188, 0.356, 0.010, 0.424, 0.334],
         [0.297, 0.454, -0.013, 0.393, 0.335],
         [0.502, 0.293, 0.001, 0.339, 0.209],
         [0.455, 0.544, 0.091, 0.471, 0.182]
         ]

# compute average specs
average_specs = [0 for _ in specs[0]]
for row in specs:
    for j in range(len(average_specs)):
        average_specs[j] += row[j]
for j in range(len(average_specs)):
    average_specs[j] /= len(average_specs)

print(average_specs)
for row in specs:
    for j in range(len(average_specs)):
        if order[j] == 1:
            row[j] = row[j] - average_specs[j]
        else:
            row[j] = average_specs[j] - row[j]


specs_mat = np.array(specs)
specs_mat = np.transpose(specs_mat)
spect_mat = list(specs_mat)
print(specs_mat)

fig, _ = graph.plot_barchart_multi_core_raw(specs_mat, color_scheme, labels, "Evaluation", "Relative score",
                                            "Evaluation results", legend, None, True, None, 0, None)

result_name = "./figs/clustering_eval_score.png"
fig.savefig(result_name, dpi=200)

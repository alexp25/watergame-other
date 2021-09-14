from sklearn import tree
import pydot
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import pandas as pd


def plot_decision_tree(mytree, feature_names, target_names, outfile):
    from dtreeplt import dtreeplt
    print("please wait.. plotting decision tree. output to: " + outfile)
    dtree = dtreeplt(
        model=mytree,
        feature_names=feature_names,
        target_names=target_names
    )   
    fig = dtree.view(interactive=False)
    # fig.set_figwidth(10)
    # fig.set_figheight(6)
    # plt.show(fig)
    #if you want save figure, use savefig method in returned figure object.
    fig.savefig(outfile)


def print_decision_tree(mytree, features, offset_unit='    '):
    '''Plots textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1, f2, f3,... if not specified
    offset_unit: a string of offset of the conditional block'''

    print("\ndecision tree:\n")

    left = mytree.tree_.children_left
    right = mytree.tree_.children_right
    threshold = mytree.tree_.threshold
    value = mytree.tree_.value
    if features is None:
        features = ['f%d' % i for i in mytree.tree_.feature]
    else:
        features = [features[i] for i in mytree.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0):
        offset = offset_unit * depth
        if threshold[node] != -2:
            print(offset + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], depth + 1)
            print(offset + "} else {")
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node], depth + 1)
            print(offset + "}")
        else:
            v = value[node][0]

            if v[0] < v[1]:
                resp = "True"
            else:
                resp = "False"

            print(offset + "return " + str(v[0]) + "/" + str(v[1]) + " => " + resp)

    recurse(left, right, threshold, features, 0, 0)


def print_tree_graph(mytree, features):
    # dot_data = StringIO()
    # tree.export_graphviz(mytree, out_file=dot_data, feature_names=features)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph = graph[0]
    # print(graph)
    tree.export_graphviz(mytree, out_file="mytree.dot", feature_names=features)
    # graph.export_png("test.png")
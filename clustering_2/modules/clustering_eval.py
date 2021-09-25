from sklearn.metrics.cluster import rand_score, adjusted_rand_score, adjusted_mutual_info_score
import numpy as np


def eval_rand_index(labels_true, labels_pred):
    rs1 = rand_score(labels_true, labels_pred)
    print(rs1)
    rs2 = adjusted_rand_score(labels_true, labels_pred)
    print(rs2)
    rs3 = adjusted_mutual_info_score(labels_true, labels_pred)
    print(rs3)

# considering same number of labels for true and pred
def eval_purity(labels_true, labels_pred):

    # https://stats.stackexchange.com/questions/95731/how-to-calculate-purity
    # Within the context of cluster analysis, Purity is an external evaluation criterion of cluster quality.
    # It is the percent of the total number of objects(data points) that were classified correctly, in the unit range [0..1].

    # build confusion matrix
    # for each cluster

    labels_dict = {}

    for i in range(len(labels_pred)):
        lps = str(labels_pred[i])
        if not lps in labels_dict:
            labels_dict[lps] = {}
        label_container = labels_dict[lps]
        lts = str(labels_true[i])
        if not lts in label_container:
            label_container[lts] = 1
        else:
            label_container[lts] += 1

    labels_mat = np.zeros((len(labels_dict), len(labels_dict)))

    # to matrix
    for k in labels_dict:
        for e in labels_dict[k]:
            labels_mat[int(k)-1][int(e)-1] = labels_dict[k][e]

    print(labels_dict)
    print(labels_mat)

    # then for each cluster ci, select the maximum value from its row, sum them together and finally divide by the total number of data points.
    s = np.shape(labels_mat)

    total_sum = np.sum(labels_mat)
    sum_max = 0
    for i in range(s[0]):
        max_i = np.max(labels_mat[i,:])
        sum_max += max_i

    p = sum_max / total_sum
    print(p)
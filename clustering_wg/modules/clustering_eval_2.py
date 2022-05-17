import numpy as np
import math
from scipy.sparse import csr_matrix


def construct_cont_table(dic):
    mat = []
    for key in dic:
        line = []
        for i in dic[key]:
            line.append(dic[key][i])
    mat = np.transpose(np.array(mat, dtype=np.int64))
    return mat

def construct_cont_table_v2(labels_true, labels_pred):

    # https://stats.stackexchange.com/questions/95731/how-to-calculate-purity
    # Within the context of cluster analysis, Purity is an external evaluation criterion of cluster quality.
    # It is the percent of the total number of objects(data points) that were classified correctly, in the unit range [0..1].

    # build confusion matrix
    # for each cluster

    #    |  T1 |  T2  |  T3
    # ---------------------
    # C1 |  0  |  53  |  10
    # C2 |  0  |  1   |  60
    # C3 |  0  |  16  |  0

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

    labels_mat = np.zeros((len(labels_pred), len(labels_true)))

    # to matrix
    for k in labels_dict:
        for e in labels_dict[k]:
            labels_mat[int(k)-1][int(e)-1] = labels_dict[k][e]

    # print(labels_dict)
    # print(labels_mat)
    return labels_mat


def rand_values(cont_table):
    n = cont_table.sum()
    sum1 = (cont_table.multiply(cont_table)).sum()
    sum2 = (np.asarray(cont_table.sum(axis=1)) ** 2).sum()
    sum3 = (np.asarray(cont_table.sum(axis=0)) ** 2).sum()
    a = (sum1 - n)/2.0
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2
    return a, b, c, d


def adj_rand_index(labels_true, labels_pred):
    mat = construct_cont_table_v2(labels_true, labels_pred)
    mat = csr_matrix(mat)
    a, b, c, d = rand_values(mat)
    nk = a+b+c+d
    return (nk*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(nk**2 - ((a+b)*(a+c) + (c+d)*(b+d)))


def rand_index(labels_true, labels_pred):
    mat = construct_cont_table_v2(labels_true, labels_pred)
    mat = csr_matrix(mat)
    a, b, c, d = rand_values(mat)
    return (a+d)/(a+b+c+d)


def calc_entropy(vector):
    h = 0.0
    # normalization
    if vector.sum() != 0:
        # normalize
        vector = vector / vector.sum()
        # remove zeros
        vector = vector[vector != 0]
        # compute h
        h = np.dot(vector, np.log2(vector) * (-1))
    return h


def entropy(labels_true, labels_pred):
    mat = construct_cont_table_v2(labels_true, labels_pred)
    h = 0.0
    n = mat.sum()
    for i in range(0, mat.shape[0]):
        h += (mat[i, :].sum() / n) * \
            (1 / math.log(mat.shape[1], 2) * calc_entropy(mat[i, :]))
    return h


def purity(labels_true, labels_pred):
    mat = construct_cont_table_v2(labels_true, labels_pred)
    n = mat.sum()
    p = 0.0
    for i in range(0, mat.shape[0]):
        p += mat[i, :].max()/n
    return p

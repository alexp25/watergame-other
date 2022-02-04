
from modules.graph import CMapMatrixElement
from typing import List
import numpy as np


def get_intersection_matrix_elems(elements: List[CMapMatrixElement], rows, cols):
    intersection_matrix = [[None for i in range(cols)] for j in range(rows)]
    for e in elements:
        intersection_matrix[e.j][e.i] = e
    return intersection_matrix


def get_intersection_matrix_vals(elements: List[CMapMatrixElement], rows, cols):
    intersection_matrix = np.zeros((rows, cols))
    for e in elements:
        intersection_matrix[e.i][e.j] = e.val
    return intersection_matrix


def get_elems_from_matrix(mat, nrows, ncols):
    elements: List[CMapMatrixElement] = []
    for row in range(nrows):
        for col in range(ncols):
            e = CMapMatrixElement()
            e.i = col
            e.j = row
            e.val = mat[row][col]
            elements.append(e)
    return elements


def check_equal_rows(row1, row2):
    eq = True
    for i in range(len(row1)):
        if row1[i] != row2[i]:
            eq = False
            break
    return eq


def write_mat_file(mat, filename, rows, cols):
    with open(filename, "w") as f:
        for i in range(rows):
            for j in range(cols):
                f.write(str(mat[i][j]) + " ")
            f.write("\n")

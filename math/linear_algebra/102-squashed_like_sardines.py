#!/usr/bin/env python3

import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    try:
        np_mat1 = np.array(mat1)
        np_mat2 = np.array(mat2)

        if np_mat1.shape[axis] != np_mat2.shape[axis]:
            return None

        return np.concatenate((np_mat1, np_mat2), axis=axis)

    except Exception as e:
        print(f"Error: {e}")
        return None

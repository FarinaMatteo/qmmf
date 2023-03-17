import os
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger("Preference Matrix Logger")
logger.setLevel("DEBUG")

def is_symmetric(A, tolerance=np.finfo(float).eps):
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A, A.transpose(), rtol=tolerance, atol=tolerance)

def is_upper_triangular(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i > j and A[i,j] != 0:
                return False
    return True


def dict_to_sample(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda x: int(x[0])))
    return np.array([v for v in sorted_dict.values()])


def make_if_needed(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder




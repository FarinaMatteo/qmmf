import cv2
import numpy as np
from .utils_sampling import localized_sampling


def get_preference_matrix_fm(x1, x2, num_of_samplings=None, threshold=0.02, sigma=6):
    """
    Samples a preference matrix on a set of homogeneous matches given by :x1: and :x2:.
    
    Params:  
    :param x1: ndarray of shape (N,3), where N is the number of matches,
    containing the coordinates in the source image;
    :param x2: identical to x1, but for the coords in the target image;
    :param num_of_samplings: (Optional) number of independent model hypotheses to build
    on the observations. Corresponds to the columns of the returned P;
    :param threshold: (Optional) inlier threshold used when sampling hypotheses. Default: 0.02.
    :param sigma: (Optional) only used if num_of_samplings is not given. 
    Determines the number of columns in P as N x sigma. Default: 6.

    Returns: P, an ndarray of shape (N, num_of_samplings) or (N, N*sigma).
    """


    assert x1.shape == x2.shape
    assert x1.ndim == 2
    assert x1.shape[1] == 3
    n_points = x1.shape[0]
    for i in range(n_points):
        assert 1 - np.finfo(float).eps <= x1[i, 2] <= 1 + np.finfo(float).eps
        assert 1 - np.finfo(float).eps <= x2[i, 2] <= 1 + np.finfo(float).eps

    preference_matrix = []
    residual_matrix = []
    
    if num_of_samplings is None: num_of_samplings = n_points*sigma


    # start sampling the models
    m = 0
    while m < num_of_samplings:
        
        # sample a MSS
        mss_indices = localized_sampling(x1, x2, k=8)
        x1_mss = x1[mss_indices, :]
        x2_mss = x2[mss_indices, :]

        # estimate the m-th F with the sampled MSS
        F, inliers_mask = cv2.findFundamentalMat(x1_mss, x2_mss, cv2.FM_RANSAC,
                                                 ransacReprojThreshold=threshold, confidence=0.99,
                                                 maxIters=100)
        if F is None: continue
        
        # compute the residuals of the estimated F
        residuals = np.ndarray(shape=(n_points))
        for i, (pt1, pt2) in enumerate(zip(x1, x2)):
            residuals[i] = cv2.sampsonDistance(pt1, pt2, F)
        
        # compute the consensus set of the estimated F
        covered_indices, = np.where(residuals < threshold)
        if covered_indices.shape[0] > 0:
            preference_column = np.zeros(shape=(n_points))
            preference_column[covered_indices] = 1
            preference_matrix.append(preference_column)
            residual_matrix.append(residuals)
            m += 1

    # The preference matrix has the points as rows, we need to compute the transpose
    preference_matrix = np.array(preference_matrix)
    return preference_matrix.T

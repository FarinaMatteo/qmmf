import numpy as np
from scipy.optimize import linear_sum_assignment


def to_point_labels(P, z):
    # define all points as outliers as the default behaviour
    point_labels = np.zeros(P.shape[0], dtype=np.int8)

    # grab all used models, then scan them to check
    # which points are covered by each model
    models = np.nonzero(z)[0]
    for i, model in enumerate(models):
        model_points = np.nonzero(P[:, model])[0]

        # mark as incorrect (i.e. -1) points which
        # are covered by more than one model
        for point in model_points:
            if point_labels[point] == 0:
                point_labels[point] = i+1
            else:
                point_labels[point] = -1
    return point_labels


def merror(P, z, z_hat, mode="onehot"):
    """Follows Appendix A of 
    https://link.springer.com/content/pdf/10.1007/s11263-021-01544-x.pdf  
    Returns: misc_error, best_label"""

    N, M = P.shape
    assert z_hat.shape[0] == M, f"Models in P and z_hat are mismatched ({M} vs {z_hat.shape[0]})."
    assert mode in ("onehot", "label")
    if mode == "label":
        assert N == z.shape[0], f"Points in P and z are mismatched ({N} vs {z.shape[0]})."
    else:
        assert M == z.shape[0], f"Models in P and z are mismatched ({M} vs {z.shape[0]})."

    # make sure domain is above 0
    if mode == "onehot":
        s = np.array(to_point_labels(P, z), dtype=np.int8) - 1
    elif mode == "label":
        s = np.array(z, dtype=np.int8) - 1
    unique_models_in_s = list(set(s.tolist()))
    if -2 in unique_models_in_s: unique_models_in_s.remove(-2)
    if -1 in unique_models_in_s: unique_models_in_s.remove(-1)
    
    # do the same for the target vector
    t = np.array(to_point_labels(P, z_hat), dtype=np.int8) - 1
    unique_models_in_t = list(set(t.tolist()))
    if -2 in unique_models_in_t: unique_models_in_t.remove(-2)
    if -1 in unique_models_in_t: unique_models_in_t.remove(-1)

    # keep track of which points are surely misclassified
    error_indices = []
    for i, label in enumerate(t):
        if label < 0:
            error_indices.append(i)
    
    # remove incorrect points from label vectors 
    # (if we keep them, we count them twice in the end)
    t_with_errors = t.copy() # needed in the last stage 
    t = np.delete(t, obj=error_indices)
    s = np.delete(s, obj=error_indices)

    # build the T and S matrices
    T = np.zeros(shape=(N-len(error_indices), len(unique_models_in_t)))
    idx = 0
    for i, label in enumerate(t):
        # skip misclassified points
        if idx < len(error_indices) and i == error_indices[idx]:
            idx += 1
            continue
        T[i, unique_models_in_t.index(label)] = 1

    S = np.zeros(shape=(N-len(error_indices), len(unique_models_in_s)))
    idx = 0
    for i, label in enumerate(s):
        # skip misclassified points
        if idx < len(error_indices) and i == error_indices[idx]:
            idx += 1
            continue
        S[i, unique_models_in_s.index(label)] = 1

    # find best permutation on classified points with the munkres alg.
    cost_matrix = -np.matmul(S.T, T)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # apply the found permutation
    col_ind = col_ind.tolist() # so we can use the 'index' method later
    permuted_t = t.copy()
    for i, elem in enumerate(t):
        # * for asymmetric problems (no. gt models != n. pred models) a match is NOT guaranteed for every structure!
        if elem in col_ind:
            permuted_t[i] = row_ind[col_ind.index(elem)]
        else:
            permuted_t[i] = elem
   
    # compute how many points are mismatched
    errors = len(error_indices)
    for gt, pred in zip(s, permuted_t):
        if gt != pred:
            errors += 1
    
    # construct the optimal permutation label
    # on the point space injecting the well-known errors
    optlabel = np.zeros(N)
    idx = 0
    for i in range(N):
        if i in error_indices:
            optlabel[i] = t_with_errors[i]
        else:
            optlabel[i] = permuted_t[idx]
            idx += 1

    return errors/N, optlabel+1

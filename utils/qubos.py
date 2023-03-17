import numpy as np
from utils.misc import is_symmetric, is_upper_triangular

def check_constraints(Q,s,logger=None):
    """
        Checks dimensionality constraints between a couplings 
        matrix :param Q: and a biases vector :param s:.
        
        In case :param logger: is provided, the umatched constraint(s), if any, is/are logged
        onto the terminal with the given logger.
    """
    ret = True
    if len(Q.shape) != 2:
        if not logger: return False
        logger.error("Q is not a matrix.")
        ret = False
    if Q.shape[0] != Q.shape[1]:
        if not logger: return False
        logger.error("Q is not a squared matrix.")
        ret = False
    if not is_symmetric(Q):
        if not logger: return False
        logger.error("Q is not a symmetric matrix.")
        ret = False
    if len(s.shape) != 1:
        if not logger: return False
        logger.error("s is not a vector.")
        ret = False
    if s.shape[0] != Q.shape[0]:
        if not logger: return False
        logger.error("number of elements in s is inconsistent with Q dimensionality.")
        ret = False
    return ret


def to_qubo_coeffs(Q, epsilon=0.001) -> dict:
    """
        Transforms a qubo problem encoded in the :param Q: matrix into a 
        QUBO dictionary feasible for Ocean's solvers. 
    """
    if not is_upper_triangular(Q):
        print("The provided Q is not an upper diagonal matrix. Cannot convert to a QUBO dict.")
        exit(0)
    qubo = {}
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            key = (i, j)
            if i == j:
                qubo[key] = Q[i,j]
            if i < j:
                if abs(Q[i,j]) > epsilon:
                    qubo[key] = Q[i,j]
    
    return qubo


def build_coef_matrix(Q, s, logger=None):
    """
        Transforms a matrix of couplings :param Q: and a vector of biases :param s:
        into an upper diagonal matrix. The values of :param s: are summed to the
        diagonal of :param Q:. 
    """
    if not check_constraints(Q,s,logger):
        logger.error("Program exiting due to inconsistent couplings matrix and biases provided.")
        exit(0)
    # zero the lower part, double the upper diagonal part
    # and add biases coming from the s vector
    S = np.zeros_like(Q)
    np.fill_diagonal(S, s)
    Q = Q.copy() + S
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if i < j:
                Q[i,j] = Q[i,j] + Q[j,i]
            elif i > j:
                Q[i,j] = 0
    return Q
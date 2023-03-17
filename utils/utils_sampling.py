import numpy as np


def get_localized_prob(pts, pt, ni):
    d_squared = np.sum(np.square(pts-pt), axis=1)
    sigma = ni * np.median(np.sqrt(d_squared))
    sigma_squared = sigma ** 2
    prob = np.exp(- (1 / sigma_squared) * d_squared)
    return prob, d_squared


def localized_sampling(src_pts, dst_pts, k, ni=1/3):
    # sample a random anchor for the MSS
    num_of_pts = src_pts.shape[0]
    g = np.random.Generator(np.random.PCG64())
    mss0 = g.choice(num_of_pts, 1)

    # grab probabilities for local sampling around the anchor
    prob_local_src, src_dists = get_localized_prob(src_pts, src_pts[mss0], ni)
    prob_local_dst, dst_dists = get_localized_prob(dst_pts, dst_pts[mss0], ni)
    prob = np.max([prob_local_src, prob_local_dst], axis=0)
    # rule out collapses around the anchor
    prob[mss0] = 0
    prob[src_dists == 0] = 0 
    prob[dst_dists == 0] = 0 
    prob = prob / np.sum(prob)

    # sample the remaining pts for the MSS
    mss1 = g.choice(num_of_pts, k-1, replace=False, p=prob)
    mss = np.array(mss0.tolist() + mss1.tolist())

    return mss
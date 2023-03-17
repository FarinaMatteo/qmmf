"""
This script runs a demo of QuMF and DeQuMF on the
MultiModel sequences of the Adelaide dataset for Fundamental Matrix
estimation.
"""
import os
import cv2
import time
import glob
import numpy as np
from scipy.io import loadmat

from problems.disjoint_set_cover import DisjointSetCover
from utils.data_generator import get_preference_matrix_fm
from utils.visualization import vis_matches
from utils.eval import merror

ADELFOLDER = "datasets/adelFM"
OUTFOLDER = "demo_output"
if not os.path.exists(OUTFOLDER): os.makedirs(OUTFOLDER)


def remove_outliers(data_dict):
    # grab the indices of outlier points
    label = data_dict["label"].squeeze()
    outlier_idxs = np.where(label == 0)[0]
    data_dict["label"] = np.delete(label, obj=outlier_idxs)
    
    # in the dataset, observations are reported as (coords, pts)
    # so a transposition is needed
    data_dict["X1_norm"] = np.delete(data_dict["X1_norm"].T, obj=outlier_idxs, axis=0)
    data_dict["X2_norm"] = np.delete(data_dict["X2_norm"].T, obj=outlier_idxs, axis=0)
    data_dict["X1"] = np.delete(data_dict["X1"].T, obj=outlier_idxs, axis=0)
    data_dict["X2"] = np.delete(data_dict["X2"].T, obj=outlier_idxs, axis=0)
    return data_dict


def sequence_name(full_path):
    bname = os.path.basename(full_path)
    seqname, ext = os.path.splitext(bname)
    return seqname


def load_adelaide(base_folder):

    files_list = sorted(glob.glob(os.path.join(base_folder, "*.mat")))
    data_dicts = [loadmat(f) for f in files_list]
    data = []
    
    for idx, data_dict in enumerate(data_dicts):

        # remove outliers as a preprocessing step
        data_dict = remove_outliers(data_dict)

        # only retain MultiModel sequences
        if len(set(data_dict["label"])) < 2: continue
        
        # only retain data useful for this demo
        data.append((
            data_dict["X1_norm"],
            data_dict["X2_norm"],
            data_dict["label"],
            data_dict["img1"],
            data_dict["img2"],
            data_dict["X1"],
            data_dict["X2"],
            sequence_name(files_list[idx])
        ))
    return data


def main():
    in_time = time.time()
    data = load_adelaide(ADELFOLDER)

    #### --- configurations for the algorithm (by default DeQuMF(SA) is set) --- ####

    ### to use DeQuMF (SA), uncomment here
    dsc = DisjointSetCover(sampler_type="sa", decompose=True)

    ### to use QuMF, uncomment here
    ### dsc = DisjointSetCover(sampler_type="qa", decompose=False)

    ### to use DeQuMF, uncomment here
    ### dsc = DisjointSetCover(sampler_type="qa", decompose=True)

    ### to use QuMF (SA), uncomment here
    ### dsc = DisjointSetCover(sampler_type="sa", decompose=False)

    ### --- additional parameters --- ###
    eps = 0.001  # use a fixed inlier threshold for all sequences

    # loop through the data in Adelaide
    errors = []
    for (X1, X2, label, img1, img2, src_coords, dst_coords, fname) in data:
        print(f"Analyzing sequence {fname}")
        # given the set of observations X1, X2, we now sample
        # a preference matrix
        P = get_preference_matrix_fm(X1, X2, threshold=eps)

        # solve the Disjoint Set Cover problem on P
        z_hat = dsc(P)

        # convert the one-hot z_hat to a structure membership vector
        err, optpred = merror(P, label, z_hat, mode="label")
        errors.append(err)
        print(f"Misclassification Rate[%] = {err*100:.2f}")

        # visualize the outcome
        out_img = vis_matches(img1, img2, src_coords, dst_coords, label=optpred)
        outpath = os.path.join(OUTFOLDER, f"{fname}.jpg")
        cv2.imwrite(outpath, out_img)
        print("Saved output at:", outpath, "\n")

    print(f"\nDemo terminated!\nYou can check qualitative results in '{OUTFOLDER}'.")
    print(f"Mean Misclassification Rate[%] = {np.array(errors).mean()*100:.2f}")
    print(f"Median Misclassification Rate[%] = {np.median(np.array(errors))*100:.2f}")
    print(f"Total time {time.time() - in_time:.2f}s")


if __name__ == "__main__":
    main()
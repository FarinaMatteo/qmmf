import cv2
import numpy as np

def vis_matches(img1, img2, X1, X2, label, draw_matches=False, stack_models=False):

    # "green", "blue", "lilla", "orange" and "yellow"
    colors = [(0,255,0), (255,0,0), (200, 160, 200), (0, 165, 255), (0, 255, 255)]
    red = (0,0,255)
    
    models_in_label = set(label.tolist())
    models_in_label.discard(0)  # classified outliers
    models_in_label.discard(-1) # overlapping among clusters

    if img1.ndim == 3 and img2.ndim == 3:
        out_img = cv2.cvtColor(cv2.cvtColor(cv2.hconcat([img1, img2]), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        out_img = cv2.cvtColor(cv2.hconcat([img1, img2]), cv2.COLOR_GRAY2BGR)
    if stack_models:
        out_img = cv2.vconcat([out_img]*len(models_in_label))
    mid_width = img1.shape[1]
    height_offset = img1.shape[0]
    pt_radius = 6

    ### INLIERS ###
    for step, (model, color) in enumerate(zip(models_in_label, colors)):
        indices = np.where(label == model)[0]
        src_pts = X1[indices, :2]
        dst_pts = X2[indices, :2]

        if not stack_models:
            step = 0

        for src_pt in src_pts:
            [x, y] = src_pt
            out_img = cv2.circle(out_img, (int(x), int(step*height_offset+y)), radius=pt_radius, color=color, thickness=-1)

        for dst_pt in dst_pts:
            [x, y] = dst_pt
            out_img = cv2.circle(out_img, (int(x+mid_width), int(step*height_offset+y)), radius=pt_radius, color=color, thickness=-1)
        
        if draw_matches:
            for src_pt, dst_pt in zip(src_pts, dst_pts):
                [src_x, src_y] = src_pt
                [dst_x, dst_y] = dst_pt
                out_img = cv2.line(out_img, (int(src_x), int(step*height_offset+src_y)), 
                                  (int(dst_x+mid_width), int(step*height_offset+dst_y)), color, thickness=1)
    

    ### OUTLIERS ###
    outliers = np.where(label == 0)[0]
    src_pts = X1[outliers, :2]
    dst_pts = X2[outliers, :2]
    for src_pt in src_pts:
        [x, y] = src_pt
        out_img = cv2.circle(out_img, (int(x), int(y)), radius=pt_radius, color=red, thickness=1)
    for dst_pt in dst_pts:
        [x, y] = dst_pt
        out_img = cv2.circle(out_img, (int(x+mid_width), int(y)), radius=pt_radius, color=red, thickness=1)

    ### OVERLAPPING POINTS ###
    outliers = np.where(label == -1)[0]
    src_pts = X1[outliers, :2]
    dst_pts = X2[outliers, :2]
    for src_pt in src_pts:
        [x, y] = src_pt
        out_img = cv2.circle(out_img, (int(x), int(y)), radius=pt_radius, color=red, thickness=-1)
    for dst_pt in dst_pts:
        [x, y] = dst_pt
        out_img = cv2.circle(out_img, (int(x+mid_width), int(y)), radius=pt_radius, color=red, thickness=-1)

    return out_img       

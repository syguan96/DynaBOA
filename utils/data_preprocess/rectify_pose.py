"""
Refer to this issue: https://github.com/akanazawa/hmr/issues/50
"""

import sys
sys.path.append('.')

import cv2
import numpy as np
import glob
import os
import os.path as osp
import pickle as pkl
from tqdm import tqdm

def rectify_pose(pose):
    """
    Rectify "upside down" people in global coord
 
    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose


if __name__=='__main__':
    mosh_path = '/data/neutrMosh/neutrSMPL_H3.6'
    mosh_files = glob.glob(osp.join(mosh_path, 'S*/*.pkl'))#'S1/Directions 1_cam0_aligned.pkl'))#'S*/*.pkl'))
    for file in tqdm(mosh_files, total=len(mosh_files)):
        if 'cam' not in file:
            continue
        mosh = pkl.load(open(file, 'rb'), encoding='latin1')
        try:
            poses = mosh['new_poses']
        except:
            import ipdb; ipdb.set_trace()
        shape = mosh['betas']
        for i in tqdm(range(poses.shape[0])):
            pose_i = rectify_pose(poses[i])
            poses[i] = pose_i
        mosh['new_poses'] = poses
        # mosh['betas'] = shape
        with open(file, 'wb') as f:
            pkl.dump(mosh,f)
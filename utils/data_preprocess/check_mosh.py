"""
Refer to this issue: https://github.com/akanazawa/hmr/issues/50
"""

import sys
sys.path.append('.')

import cv2
import glob
import torch
import config
import numpy as np
import pickle as pkl
from utils.smpl import SMPL


def scatter_pose(image, joints):
    #import ipdb
    #ipdb.set_trace()
    joints = (joints + 1) / 2 * 500 + 500
    for idx in range(joints.shape[0]):
        pt = joints[idx]
        cv2.circle(image, (pt[0], pt[1]), 4, np.array([197, 27, 125]).tolist(), -1)
    cv2.imwrite('tmp.jpg', image)        


if __name__ == '__main__':
    idx = 0
    mosh_file = '/data/syguan/human_datasets/neutrMosh/neutrSMPL_H3.6/S5/Directions 1_cam1_aligned.pkl'
    images = '/data/syguan/human_datasets/human36m/images/S5_Directions_1.54138969_{:06d}.jpg'.format(idx+1)

    mosh = pkl.load(open(mosh_file, 'rb'), encoding='latin1')
    pose = mosh['new_poses'][idx]
    shape = mosh['betas']

    pose = torch.from_numpy(pose).float().unsqueeze(0)
    shape = torch.from_numpy(shape).float().unsqueeze(0)

    smpl = SMPL(config.SMPL_MODEL_DIR,
                    batch_size=1, 
                    create_transl=False).cpu()

    smpl_out = smpl(betas=shape, body_pose=pose[:,3:], global_orient=pose[:, :3])
    vertices = smpl_out.vertices
    joints = smpl_out.joints

    scatter_pose(cv2.imread(images), joints[0])

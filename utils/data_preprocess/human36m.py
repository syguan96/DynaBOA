import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
from spacepy import pycdf
from torch import use_deterministic_algorithms
from tqdm import tqdm
import pickle as pkl
from utils.kp_utils import get_perm_idxs

CAMERA_DICT = {
    '55011271': 'cam1',
    '58860488': 'cam2',
    '60457274': 'cam3',
    '54138969': 'cam0',
}

DEBUG = False

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, training_split=True):

    # users in validation set
    if training_split:
        user_list = [1, 5, 6, 7, 8]
    else:
        user_list = [9, 11]

    # go over each user
    for user_i in tqdm(user_list, desc='user'):
        user_name = 'S%d' % user_i
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in tqdm(seq_list, desc='seq'):
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # video file
            vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            imgs_path = os.path.join(dataset_path, 'images')
            vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                success, image = vidcap.read()
                if not success:
                    break

                # check if you can keep this frame
                if frame_i % 5 == 0 and camera == '60457274': #and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    img_out = os.path.join(imgs_path, imgname)
                    cv2.imwrite(img_out, image)
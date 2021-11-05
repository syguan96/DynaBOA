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
def h36m_train_extract(dataset_path, out_path, training_split=True, extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    # global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    global_idx = [i-25 for i in get_perm_idxs(src='spin', dst='h36m')]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_  = [], [], [], [], []
    poses_, shapes_ = [], []

    # users in validation set
    if training_split:
        user_list = [1, 5, 6, 7, 8]
    else:
        user_list = [9, 11]

    # go over each user
    for user_i in tqdm(user_list, desc='user'):
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')
        # mosh path
        mosh_path = os.path.join('/data/syguan/human_datasets/neutrMosh/neutrSMPL_H3.6', user_name)

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in tqdm(seq_list, desc='seq'):

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action_raw = action
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # mosh file， which has been downsampled to 10 fps
            mosh_file = os.path.join(mosh_path, '_'.join([action_raw, CAMERA_DICT[camera], 'aligned.pkl']))
            try:
                mosh = pkl.load(open(mosh_file, 'rb'), encoding='latin1')
            except:
                print('mosh file: {} does not exist!'.format(mosh_file))
                continue
            pose = mosh['new_poses']
            shape = mosh['betas']
            mosh_idx = 0
            if DEBUG:
                import ipdb
                ipdb.set_trace()

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 5 == 0 and camera == '60457274': #and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)
                    
                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24,3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    # read mosh
                    pose_i = pose[mosh_idx]
                    mosh_idx += 1

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S24)
                    poses_.append(pose_i)
                    shapes_.append(shape)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    if training_split:
        out_file = os.path.join(out_path, 'h36m_train.npz')
    else:
        out_file = os.path.join(out_path, 'h36m_valid.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       part=parts_,
                       S=Ss_,)

def h36m_eval_extract(dataset_path, out_path, protocol=1, extract_img=True):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_  = [], [], [], [], []

    # users in validation set
    user_list = [9, 11]

    # go over each user
    for user_i in tqdm(user_list):
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'test_images')
                vidcap = cv2.VideoCapture(vid_file)
                success, image = vidcap.read()

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24,3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    Ss_.append(S24)
                    parts_.append(part)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 
        'h36m_valid_protocol%d.npz' % protocol)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_)

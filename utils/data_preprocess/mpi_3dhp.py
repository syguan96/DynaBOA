import os
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import imageio

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

        
def test_data(dataset_path, out_path, joints_idx, scaleFactor):

    joints17_idx = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]

    # imgnames_, scales_, centers_, parts_,  Ss_ = [], [], [], [], []

    # training data
    user_list = range(1,7)

    for user_i in tqdm(user_list):
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])
        imgnames_, scales_, centers_, parts_,  Ss_ = [], [], [], [], []
        for frame_i, valid_i in enumerate(valid):
            if valid_i == 0:
                continue
            img_name = os.path.join('mpi_inf_3dhp_test_set',
                                   'TS' + str(user_i),
                                   'imageSequence',
                                   'img_' + str(frame_i+1).zfill(6) + '.jpg')

            joints = annot2[frame_i,0,joints17_idx,:]
            S17 = annot3[frame_i,0,joints17_idx,:]/1000
            S17 = S17 - S17[0]

            bbox = [min(joints[:,0]), min(joints[:,1]),
                    max(joints[:,0]), max(joints[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_name)
            # I = scipy.misc.imread(img_file)
            I = imageio.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < len(joints_idx):
                continue

            part = np.zeros([24,3])
            part[joints_idx] = np.hstack([joints, np.ones([17,1])])

            S = np.zeros([24,4])
            S[joints_idx] = np.hstack([S17, np.ones([17,1])])

            # store the data
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            Ss_.append(S)
        out_file = os.path.join(out_path, f'3dhp_{user_i}.npz')
        np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_) 

    # # store the data struct
    # if not os.path.isdir(out_path):
    #     os.makedirs(out_path)
    # out_file = os.path.join(out_path, 'mpi_inf_3dhp_nouni_test.npz')
    # np.savez(out_file, imgname=imgnames_,
    #                    center=centers_,
    #                    scale=scales_,
    #                    part=parts_,
    #                    S=Ss_)    

def mpi_inf_3dhp_extract(dataset_path, out_path, mode, extract_img=False, static_fits=None):

    scaleFactor = 1 #1.2
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    
    if static_fits is not None:
        fits_3d = os.path.join(static_fits, 
                               'mpi-inf-3dhp_mview_fits.npz')
    else:
        fits_3d = None
    
    test_data(dataset_path, out_path, joints_idx, scaleFactor)

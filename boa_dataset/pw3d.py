"""
Mix the video from different datasets
"""

import os
import os.path as osp
import cv2
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import constants
import config
from utils.dataprocess import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

def key_3dpw(elem):
    elem = os.path.basename(elem)
    vid = elem.split('_')[1]
    pid = elem.split('_')[2][:-4]
    return int(vid)*10+int(pid)


class PW3D(Dataset):
    def __init__(self, options):
        super(PW3D, self).__init__()
        self.options = options
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        
        # 3DPW
        self.pw3d_img_dir = config.PW3D_ROOT
        self.pw3d_datas = glob.glob('data/dataset_extras/3dpw_[0-9]*_[0-9].npz')
        self.pw3d_datas.sort(key=key_3dpw)
        print(f'pw3d: {len(self.pw3d_datas)}')

        # parse data
        self.datas = []
        self.imgnames = []
        self.scales = []
        self.centers = []
        self.smpl_j2ds = []
        self.genders = []
        self.dataset_names = []
        # for 3dhp
        self.gt_j3ds = []
        # for 3dpw
        self.pose, self.betas, self.smpl_j2ds, self.op_j2ds = [], [], [], []

        for i in range(len(self.pw3d_datas)):
            self.datas.append(self.pw3d_datas[i])
            imgnames, scales, centers, pose, betas, smpl_j2ds, op_j2ds, genders, gt_j3ds, dataset_name = self.parse_3dpw(self.pw3d_datas[i])            
            
            self.imgnames.append(imgnames)
            self.scales.append(scales)
            self.centers.append(centers)
            self.smpl_j2ds.append(smpl_j2ds)
            self.genders.append(genders)
            self.gt_j3ds.append(gt_j3ds)
            self.pose.append(pose)
            self.betas.append(betas)
            self.op_j2ds.append(op_j2ds)
            self.dataset_names.append(dataset_name)
        assert len(self.datas) == len(self.pw3d_datas)
        # record it 
        with open(osp.join(self.options.expdir, self.options.expname, 'seq_order.record'), 'w') as f:
            for data_name in self.datas:
                f.write(data_name+'\n')
        
        self.imgnames = np.concatenate(self.imgnames, axis=0)
        self.scales = np.concatenate(self.scales, axis=0)
        self.centers = np.concatenate(self.centers, axis=0)
        self.smpl_j2ds = np.concatenate(self.smpl_j2ds, axis=0)
        self.genders = np.concatenate(self.genders, axis=0)
        self.gt_j3ds = np.concatenate(self.gt_j3ds, axis=0)
        self.pose = np.concatenate(self.pose, axis=0)
        self.betas = np.concatenate(self.betas, axis=0)
        self.op_j2ds = np.concatenate(self.op_j2ds, axis=0)
        self.dataset_names = np.concatenate(self.dataset_names, axis=0)

    def __len__(self):
        return self.scales.shape[0]


    def __getitem__(self, index):
        item = {}

        scale = self.scales[index].copy()
        center = self.centers[index].copy()
        op_j2d = self.op_j2ds[index].copy()
        theta = self.pose[index].copy()
        beta = self.betas[index].copy()
        imgname = self.imgnames[index].copy()
        smpl_j2d = self.smpl_j2ds[index].copy()
        gender = self.genders[index].copy()
        dataset_name = self.dataset_names[index].copy()
        j3d = self.gt_j3ds[index].copy()

        image = self.read_image(imgname, index)

        # ori image, no aug
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        op_j2d, image, theta, beta, smpl_j2d = self.process_sample(image.copy(),
                                                                          theta, 
                                                                          beta, 
                                                                          op_j2d, 
                                                                          smpl_j2d, 
                                                                          center, 
                                                                          scale, 
                                                                          flip, pn, rot, sc, is_train=False)
        item['op_j2d'] = op_j2d
        item['image'] = image
        item['pose'] = theta
        item['betas'] = beta
        item['smpl_j2d'] = smpl_j2d
        item['gender'] = gender
        item['imgname'] = imgname
        item['dataset_name'] = dataset_name
        item['j3d'] = j3d
        item['bbox'] = np.stack([center[0], center[1], scale * 200])
        return item

    def process_sample(self, image, pose, beta, keypoints, smpl_j2ds, center, scale, flip, pn, rot, sc, is_train):
        # labeled keypoints
        kp2d = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip, is_train=is_train)).float()
        smpl_j2ds = torch.from_numpy(self.j2d_processing(smpl_j2ds, center, sc*scale, rot, flip, is_train=is_train)).float()
        img = self.rgb_processing(image, center, sc*scale, rot, flip, pn, is_train=is_train)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)
        pose = torch.from_numpy(self.pose_processing(pose, rot, flip, is_train=is_train)).float()
        betas = torch.from_numpy(beta).float()
        return kp2d, img, pose, betas, smpl_j2ds

    def read_image(self, imgname, index):
        if self.dataset_names[index] == '3dpw':
            imgname = os.path.join(self.pw3d_img_dir, imgname)
        elif self.dataset_names[index] == '3dhp':
            imgname = os.path.join(self.hp3d_img_dir, imgname)
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        return img

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, is_train):
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        if is_train and flip:
                rgb_img = flip_img(rgb_img)
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_train):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if is_train and f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def pose_processing(self, pose, r, f, is_train):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        if is_train:
            # rotation or the pose parameters
            pose[:3] = rot_aa(pose[:3], r)
            # flip the pose parameters
            if f:
                pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def parse_3dpw(self, dataname):
        data = np.load(dataname)
        imgnames = data['imgname']
        scales = data['scale']
        centers = data['center']
        pose = data['pose'].astype(np.float)
        betas = data['shape'].astype(np.float)
        smpl_j2ds = data['j2d']
        op_j2ds = data['op_j2d']
        # Get gender data, if available
        try:
            gender = data['gender']
            genders = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            genders = -1*np.ones(len(imgnames)).astype(np.int32)
        gt_j3ds = np.zeros((scales.shape[0], 24, 4))
        dataset_name =  ['3dpw'] * scales.shape[0]
        return imgnames, scales, centers, pose, betas, smpl_j2ds, op_j2ds, genders, gt_j3ds, dataset_name
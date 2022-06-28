
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
from utils.dataprocess import crop, flip_img, flip_kp, transform

class Internet_dataset(Dataset):
    def __init__(self):
        super(Internet_dataset, self).__init__()
        self.imgdir = osp.join(config.InternetData_ROOT, 'images')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        datanames = glob.glob(osp.join(config.InternetData_ROOT, '*.npz'))
        self.smpl_j2ds = []
        self.imgnames = []
        self.scales = []
        self.centers = []
        print(len(datanames))
        for dataname in datanames:
            # load saved data
            data = np.load(dataname)
            self.imgnames.append(data['imgname'])
            print(len(data['imgname']))
            self.scales.append(data['scale'])
            self.centers.append(data['center'])
            self.smpl_j2ds.append(data['part'])
        self.imgnames = np.concatenate(self.imgnames, 0)
        self.scales = np.concatenate(self.scales, 0)
        self.centers = np.concatenate(self.centers, 0)
        self.smpl_j2ds = np.concatenate(self.smpl_j2ds, 0)
        print('LEN:', self.imgnames.shape[0])


    def __getitem__(self, index):
        item = {}
        scale = self.scales[index]
        center = self.centers[index]
        imgname = self.imgnames[index]
        smpl_j2d = self.smpl_j2ds[index]

        image = self.read_image(imgname)
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        img, smpl_j2d = self.process_sample(image, smpl_j2d, center, scale, flip, pn, rot, sc, False)
        item['image'] = img
        item['imgname'] = imgname
        item['smpl_j2d'] = smpl_j2d
        item['bbox'] = np.stack([center[0], center[1], scale * 200])
        return item

    def __len__(self):
        return self.scales.shape[0]

    def read_image(self, imgname):
        imgname = os.path.join(self.imgdir, imgname)
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        return img


    def process_sample(self, image, smpl_j2d, center, scale, flip, pn, rot, sc, is_train):
        # labeled keypoints
        smpl_j2d = torch.from_numpy(self.j2d_processing(smpl_j2d, center, sc*scale, rot, flip, is_train=is_train)).float()
        img = self.rgb_processing(image, center, sc*scale, rot, flip, pn, is_train=is_train)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)
        return img, smpl_j2d

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

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, is_train):
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        if is_train and flip:
                rgb_img = flip_img(rgb_img)
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

"""
Support Webcam
"""
from glob import glob
from operator import gt
from selectors import EpollSelector
import shutil
import sys

sys.path.append('..')

import os
import cv2
import random
import argparse
import numpy as np
import learn2learn as l2l

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

import constants
import config
from model import SMPL, hmr
from utils.smplify.prior import MaxMixturePrior
from utils.geometry import batch_rodrigues, perspective_projection, rotation_matrix_to_angle_axis
from utils.kp_utils import get_perm_idxs
from utils.dataprocess import crop, flip_img, flip_kp, transform
from vid2img import video_to_images

from utils.webcam_utils import WebcamVideoStream, OpenposeWarper, render


class Adaptor():
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda')
        self.seed_everything(self.options.seed)

        self._initialize_training()

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def _initialize_training(self):
        self.history = {} 
        self.global_step = 0
        # build model
        checkpoint = torch.load(self.options.model_file)
        model = hmr(config.SMPL_MEAN_PARAMS)
        if self.options.use_boa:
            self.model = l2l.algorithms.MAML(model, lr=self.options.fastlr, first_order=True).to(self.device)
            self.model.load_state_dict(checkpoint['model'], strict=True)
        else:
            self.model = model.to(self.device)
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        
        # build teacher model
        if self.options.use_meanteacher:
            model = hmr(config.SMPL_MEAN_PARAMS)
            for param in model.parameters():
                param.detach_()
            self.teacher = model.to(self.device)
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            self.teacher.load_state_dict(checkpoint['model'], strict=True)
            self.teacher.eval()

        if self.options.test_basemodel:
            self.basemodel = hmr(config.SMPL_MEAN_PARAMS).to(self.device)
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            self.basemodel.load_state_dict(checkpoint['model'], strict=True)

        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.lr, betas=(self.options.beta1, self.options.beta2))

        # build criterions
        self.gmm_f = MaxMixturePrior(prior_folder='data/', num_gaussians=8, dtype=torch.float32).to(self.device)

        # build SMPL model, here we use the neutral model
        self.smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(self.device)
    
    # Below is the utils of the adaptor
    def seed_everything(self, seed):
        """
        ensure reproduction
        """
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

    def get_hist(self,):
        infos = self.history[self.global_step - self.options.interval]
        return torch.from_numpy(infos['image']).to(self.device), torch.from_numpy(infos['s2d']).to(self.device)

    def save_hist(self, image, s2d):
        self.history[self.global_step] = {'image': image.detach().cpu().numpy(), 
                                          's2d': s2d.detach().cpu().numpy()}
        self.global_step += 1
    
    def decode_smpl_params(self, poses, beta, pose2rot=False):
        smpl_out = self.smpl_neutral(betas=beta, body_pose=poses[:,1:], global_orient=poses[:,0].unsqueeze(1), pose2rot=pose2rot)
        return {'s3d': smpl_out.joints, 'vts': smpl_out.vertices}

    def update_teacher(self, teacher, model):
        """
        teacher = teacher * alpha + model * (1 - alpha)
        """
        factor = self.options.alpha
        for param_t, param in zip(teacher.parameters(), model.parameters()):
            param_t.data.mul_(factor).add_(param.data, alpha=1 - factor)

    def cal_feature_diff(self, features_i, features_j):
        sim = F.cosine_similarity(features_i[12].flatten(), features_j[12].flatten(), dim=0, eps=1e-12)
        return sim.item()

    def projection(self, cam, s3d, eps=1e-9):
        cam_t = torch.stack([cam[:,1], cam[:,2],
                            2*constants.FOCAL_LENGTH/(constants.IMG_RES * cam[:,0] + eps)],dim=-1)
        camera_center = torch.zeros(s3d.shape[0], 2, device=self.device)
        s2d = perspective_projection(s3d,
                                    rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(s3d.shape[0], -1, -1),
                                    translation=cam_t,
                                    focal_length=constants.FOCAL_LENGTH,
                                    camera_center=camera_center)
        s2d_norm = s2d / (constants.IMG_RES / 2.)  # to [-1,1]
        return {'ori':s2d, 'normed': s2d_norm}

    def cal_shape_prior(self, pred_betas):
        return (pred_betas**2).sum(dim=-1).mean()
    
    def cal_pose_prior(self, pred_rotmat, betas):
        # gmm prior
        body_pose = rotation_matrix_to_angle_axis(pred_rotmat[:,1:].contiguous().view(-1,3,3)).contiguous().view(-1, 69)
        pose_prior_loss = self.gmm_f(body_pose, betas).mean()
        return pose_prior_loss
    
    def cal_teacher_loss(self, image, pred_rotmat, pred_shape, pred_s2d, pred_s3d):
        """
        we calculate same loss items as SPIN. 
        """
        ema_rotmat, ema_shape, ema_cam = self.teacher(image)
        ema_smpl_out = self.decode_smpl_params(ema_rotmat, ema_shape)
        ema_pred_s3d = ema_smpl_out['s3d']
        ema_pred_vts = ema_smpl_out['vts']

        # 2d and 3d kp losses
        ema_s2d = self.projection(ema_cam, ema_pred_s3d)['normed']
        s2dloss = F.mse_loss(pred_s2d, ema_s2d)
        s3dloss = F.mse_loss(ema_pred_s3d, pred_s3d)
        # beta and theta losses
        shape_loss = F.mse_loss(pred_shape, ema_shape)
        pose_loss = F.mse_loss(pred_rotmat, ema_rotmat)

        loss = s2dloss * 5 + s3dloss * 5 + shape_loss * 0.001 + pose_loss * 1
        return loss

    def cal_motion_loss(self, model, pred_s2d, gt_s2d, prefix='ul'):
        hist_image, hist_s2d = self.get_hist()
        hist_pred_rotmat, hist_pred_shape, hist_pred_cam = model(hist_image)
        hist_smpl_out = self.decode_smpl_params(hist_pred_rotmat, hist_pred_shape)
        hist_pred_s3d = hist_smpl_out['s3d']

        # cal motion loss
        hist_pred_s2d = self.projection(hist_pred_cam, hist_pred_s3d)['normed']
        pred_motion = pred_s2d - hist_pred_s2d[:,:25]
        gt_motion = gt_s2d[:,:,:-1] - hist_s2d[:,:,:-1]
        # cal non-zero confidence
        conf1 = hist_s2d[:,:, -1].unsqueeze(-1).clone()
        conf2 = gt_s2d[:,:, -1].unsqueeze(-1).clone()
        one = torch.tensor([1.]).to(self.device)
        zero = torch.tensor([0.]).to(self.device)
        conf = torch.where((conf1 + conf2)==2,one,zero)
        
        motion_loss = (F.mse_loss(pred_motion, gt_motion, reduction='none')*conf).mean()
        return motion_loss

    def reload(self):
        checkpoint = torch.load(self.options.model_file)
        if self.options.use_boa:
            self.model.load_state_dict(checkpoint['model'], strict=True)
        else:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            self.model.load_state_dict(checkpoint['model'], strict=True)

        if self.options.use_meanteacher:
            self.teacher.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.lr, betas=(self.options.beta1, self.options.beta2))
        print('the adaptor is reset')

    def dataprocess(self, image, gtkp2d, scaleFactor=1.0):
        # calcualte boundingbox
        bbox = [min(gtkp2d[:,0]), min(gtkp2d[:,1]),
                        max(gtkp2d[:,0]), max(gtkp2d[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
        bbox = np.stack([center[0], center[1], scale * 200])

        # process 2D keypoints
        gtkp2d[:,2] = gtkp2d[:,2] > 0.3
        for i in range(gtkp2d.shape[0]):
            gtkp2d[i,0:2] = transform(gtkp2d[i,0:2].copy()+1, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=0)
        gtkp2d[:,:-1] = 2.*gtkp2d[:,:-1]/constants.IMG_RES - 1.
        gtkp2d = torch.from_numpy(gtkp2d.astype('float32')).float()

        # process image
        image = crop(image.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=0)
        image = np.transpose(image.astype('float32'),(2,0,1))/255.0
        image = torch.from_numpy(image).float()
        image = self.normalize_img(image)
        return image.unsqueeze(0).to(self.device), gtkp2d.unsqueeze(0).to(self.device), bbox[None,:]

    # adaptaion and inference
    def online_adaptation(self, ori_image, gtkp2d):
        # placeholder for processing image and gtkp2d
        image, gt_keypoints_2d, bbox = self.dataprocess(ori_image, gtkp2d[0].copy(), scaleFactor=1.2)
        self.save_hist(image, gt_keypoints_2d)

        # adaptation
        if not self.options.use_boa:
            pred_rotmat, pred_shape, pred_cam = self.model(image, need_feature=False)
            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_s3d = smpl_out['s3d']
            pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
            conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
            # re-projection loss
            s2dloss = (F.mse_loss(pred_s2d[:, :25], gt_keypoints_2d[:, :, :-1], reduction='none')*conf).mean()
            # shape prior constraint
            shape_prior = self.cal_shape_prior(pred_shape)
            # pose prior constraint
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)
            loss = s2dloss * self.options.s2dloss_weight +\
                            shape_prior * self.options.shape_prior_weight +\
                            pose_prior * self.options.pose_prior_weight
            # update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            # clone the intial model to calculate similarity
            with torch.no_grad():
                _,_,_, init_features = self.model(image, need_feature=True)
            # lower-level adaptation
            learner = self.model.clone()
            pred_rotmat, pred_shape, pred_cam = learner(image, need_feature=False)
            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_s3d = smpl_out['s3d']
            pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
            conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
            s2dloss = (F.mse_loss(pred_s2d[:, :25], gt_keypoints_2d[:, :, :-1], reduction='none')*conf).mean()
            shape_prior = self.cal_shape_prior(pred_shape)
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)
            lowerlevel_loss = s2dloss * self.options.s2dloss_weight +\
                            shape_prior * self.options.shape_prior_weight +\
                            pose_prior * self.options.pose_prior_weight
            learner.adapt(lowerlevel_loss)

            # upper-level adaptation
            pred_rotmat, pred_shape, pred_cam = learner(image, need_feature=False)
            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_s3d = smpl_out['s3d']
            pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
            conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
            s2dloss = (F.mse_loss(pred_s2d[:, :25], gt_keypoints_2d[:, :, :-1], reduction='none')*conf).mean()
            shape_prior = self.cal_shape_prior(pred_shape)
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)
            upperlevel_loss = s2dloss * self.options.s2dloss_weight +\
                            shape_prior * self.options.shape_prior_weight +\
                            pose_prior * self.options.pose_prior_weight
            if self.options.use_motion and (self.global_step - self.options.interval) > 0:
                motionloss = self.cal_motion_loss(learner, pred_s2d[:, :25], gt_keypoints_2d, prefix='ul')
                upperlevel_loss = upperlevel_loss + motionloss * self.options.motionloss_weight
            if self.options.use_meanteacher:
                teacherloss = self.cal_teacher_loss(image, pred_rotmat, pred_shape, pred_s2d, pred_s3d)
                upperlevel_loss = upperlevel_loss + teacherloss * self.options.teacherloss_weight
            
            self.optimizer.zero_grad()
            upperlevel_loss.backward()
            self.optimizer.step()
            self.update_teacher(self.teacher, self.model)

            if self.options.dynamic_boa:
                _,_,_, adapted_features = self.model(image, need_feature=True)
                feat_12 = self.cal_feature_diff(init_features, adapted_features)
                self.optimized_step = 0
                while 1-feat_12 > self.options.cos_sim_threshold:
                    self.optimized_step += 1
                    if self.optimized_step > self.options.optim_steps:
                        break
                    pred_rotmat, pred_shape, pred_cam = self.model(image, need_feature=False)
                    smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
                    pred_s3d = smpl_out['s3d']
                    pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
                    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
                    s2dloss = (F.mse_loss(pred_s2d[:, :25], gt_keypoints_2d[:, :, :-1], reduction='none')*conf).mean()
                    shape_prior = self.cal_shape_prior(pred_shape)
                    pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)
                    upperlevel_loss = s2dloss * self.options.s2dloss_weight +\
                                    shape_prior * self.options.shape_prior_weight +\
                                    pose_prior * self.options.pose_prior_weight
                    if self.options.use_motion and (self.global_step - self.options.interval) > 0:
                        motionloss = self.cal_motion_loss(self.model, pred_s2d[:, :25], gt_keypoints_2d, prefix='ul')
                        upperlevel_loss = upperlevel_loss + motionloss * self.options.motionloss_weight
                    if self.options.use_meanteacher:
                        teacherloss = self.cal_teacher_loss(image, pred_rotmat, pred_shape, pred_s2d, pred_s3d)
                        upperlevel_loss = upperlevel_loss + teacherloss * self.options.teacherloss_weight
                    # upper_level_loss, adapted_features = self.upper_level_adaptation(image, gt_keypoints_2d, h36m_batch, self.model)
                    self.optimizer.zero_grad()
                    upperlevel_loss.backward()
                    self.optimizer.step()
                    if self.options.use_meanteacher:
                        self.update_teacher(self.teacher, self.model)
                    with torch.no_grad():
                        init_features = adapted_features
                        _,_,_, adapted_features = self.model(image, need_feature=True)
                        feat_12 = self.cal_feature_diff(init_features, adapted_features)
        # inference
        pred_rotmat, pred_shape, pred_cam = self.model(image, need_feature=False)
        smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
        pred_vertices = smpl_out['vts']
        res = {'vts': pred_vertices, 'cam': pred_cam, 'bbox': bbox}

        # use basemodel to estimate smpl paramerters
        if self.options.test_basemodel:
            pred_rotmat_base, pred_shape_base, pred_cam_base = self.basemodel(image, need_feature=False)
            smpl_out_base = self.decode_smpl_params(pred_rotmat_base, pred_shape_base)
            pred_vertices_base = smpl_out_base['vts']
            res_base = {'vts_base': pred_vertices_base, 'cam_base': pred_cam_base}
            res.update(res_base)

        return res

parser = argparse.ArgumentParser()
parser.add_argument('--capture_mode', type=str, default='webcam', choices=['webcam', 'video'])
parser.add_argument('--vid_path', type=str, default=None, help='video path if capture mode is video')
parser.add_argument('--test_basemodel', type=int, default=0, help='whether to test the basemodel')

parser.add_argument('--save_video', type=int, default=0)
parser.add_argument('--res_dir', type=str, default='temp')

parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument('--model_file', type=str, default='/data/mesh_reconstruction/DynaBOA/data/basemodel.pt')
parser.add_argument('--lr', type=float, default=3e-6, help='learning rate of the upper-level')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of adam')

parser.add_argument('--use_boa', type=int, default=0)
parser.add_argument('--fastlr', type=float, default=8e-6, help='fast learning rate, which is the parameter of lower-level')
parser.add_argument('--s2dloss_weight', type=float, default=10)
parser.add_argument('--shape_prior_weight', type=float, default=2e-6)
parser.add_argument('--pose_prior_weight', type=float, default=1e-4)

# teacher
parser.add_argument('--use_meanteacher', type=int, default=1,choices=[0,1], help='1: use mean teacher')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha * teacher + (1-alpha) * model, scale: 0-1')
parser.add_argument('--teacherloss_weight', type=float, default=0.1)

# motion
parser.add_argument('--use_motion', type=int, default=1,choices=[0,1], help='1: use mean teacher')
parser.add_argument('--interval', type=int, default=5, help='interval of temporal loss, scale: >= 1')
parser.add_argument('--motionloss_weight', type=float, default=0.8)

parser.add_argument('--dynamic_boa', type=int, default=0, choices=[0,1], help='dynamic boa')
parser.add_argument('--cos_sim_threshold', type=float, default=3.1e-4, help='cos sim threshold')
parser.add_argument('--optim_steps', type=int, default=7, help='steps of the boa for the current image')

if __name__ == '__main__':
    from time import time
    options = parser.parse_args()

    save_video = True
    videoname = 'output.mp4'
    frame_rate = 10.0

    openpose_estimator = OpenposeWarper()
    adaptor = Adaptor(options)

    if options.capture_mode == 'webcam':
        cap =  WebcamVideoStream(0)
        cap.start()
    elif options.capture_mode == 'video':
        images_folder = video_to_images(options.vid_path, img_folder='image_caches')
        image_paths = glob(os.path.join(images_folder, '*.png'))
        image_paths.sort(key=lambda x: int(x.split('/')[-1][:-4]), reverse=True)

    frame_idx = 0
    while True:
        if options.capture_mode == 'webcam':
            frame = cap.read()
        elif options.capture_mode == 'video':
            image_path = image_paths.pop()
            print(image_path) # for debug
            frame = cv2.imread(image_path)

        # use openpose to estiamte 2D keypoints
        kp2d, annoted_image = openpose_estimator.estimate(frame)

        try:
            # online adaptation
            res = adaptor.online_adaptation(frame[:,:,::-1].copy(), kp2d)
            vertices, pred_cam, bbox = res['vts'], res['cam'], res['bbox']

            # rendering mesh
            rendered_image = render(vertices, pred_cam, frame[:,:,::-1].copy(), bbox)

            final_image = np.concatenate([annoted_image, rendered_image], axis=1)

            if options.test_basemodel:
                vertices_base, pred_cam_base = res['vts_base'], res['cam_base']
                rendered_image_base = render(vertices_base, pred_cam_base, frame[:,:,::-1].copy(), bbox, color=[100,100,200])

                final_image = np.concatenate([final_image, rendered_image_base], axis=1)

        except TypeError:
            if options.test_basemodel:
                final_image = np.concatenate([annoted_image, frame, frame], axis=1)
            else:
                final_image = np.concatenate([annoted_image, frame], axis=1)

        if options.save_video:
            if frame_idx == 0:
                os.makedirs(options.res_dir, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(os.path.join(options.res_dir,'output.mp4'),fourcc, frame_rate,(final_image.shape[1], final_image.shape[0]))
            out.write(final_image)
            frame_idx = 1
        
        # cv2.imshow('webcam', frame)
        cv2.imshow('openpose', final_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if options.capture_mode == 'webcam':
                cap.stop()
            elif options.capture_mode == 'video':
                shutil.rmtree('image_caches')
            break
        if cv2.waitKey(1) & 0xFF == ord('r'):
            adaptor.reload()

    cv2.destroyAllWindows()
    if options.save_video:
        out.release()
        

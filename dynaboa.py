"""
boa
seperate version, i.e. the batch in inner loop is different from the outer loop.
more than that, we reweight the loss weight for the boa. 
The reweight scheme is inspired from the focal loss.
"""

import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import random
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import learn2learn as l2l


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

import config
import constants
from model import SMPL, hmr
from utils.smplify.prior import MaxMixturePrior
from utils.geometry import batch_rodrigues, perspective_projection, rotation_matrix_to_angle_axis
from utils.pose_utils import compute_similarity_transform_batch
from boa_dataset.pw3d import PW3D
from render_demo import Renderer, convert_crop_cam_to_orig_img


parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, default='exps')
parser.add_argument('--expname', type=str, default='3dpw')
parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument('--seq_seed', type=int, default=22, help='random seed')
parser.add_argument('--model_file', type=str, default='data/basemodel.pt')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--save_res', type=int, default=0, choices=[0,1], help='save middle mesh and image')

parser.add_argument('--lr', type=float, default=3e-6, help='learning rate of the upper-level')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of adam')

# boa
parser.add_argument('--use_boa', type=int, default=1, choices=[0,1], help='use boa')
parser.add_argument('--fastlr', type=float, default=8e-6, help='fast learning rate, which is the parameter of lower-level')
parser.add_argument('--inner_step', type=int, default=1, help='steps of inner loop')
parser.add_argument('--record_lowerlevel', type=int, default=1, help='record results of the lowerelevel?')
parser.add_argument('--s2dloss_weight', type=float, default=10)
parser.add_argument('--shape_prior_weight', type=float, default=2e-6)
parser.add_argument('--pose_prior_weight', type=float, default=1e-4)

parser.add_argument('--use_frame_losses_lower', type=int, default=1, choices=[0,1], help='whether use frame-wise losses at lower level')
parser.add_argument('--use_frame_losses_upper', type=int, default=1, choices=[0,1], help='whether use frame-wise losses at upper level')
parser.add_argument('--use_temporal_losses_lower', type=int, default=0, choices=[0,1], help='whether use temporal-wise losses at lower level')
parser.add_argument('--use_temporal_losses_upper', type=int, default=1, choices=[0,1], help='whether use temporal-wise losses at upper level')

parser.add_argument('--sample_num', type=int, default=1, help='sample_num')
parser.add_argument('--retrieval', type=int, default=1, choices=[0,1], help='use retrieval')

parser.add_argument('--dynamic_boa', type=int, default=1, choices=[0,1], help='dynamic boa')
parser.add_argument('--cos_sim_threshold', type=float, default=0.002, help='cos sim threshold')
parser.add_argument('--optim_steps', type=int, default=7, help='steps of the boa for the current image')

# mix training
parser.add_argument('--lower_level_mixtrain', type=int, default=1, choices=[0,1], help='use mix training')
parser.add_argument('--upper_level_mixtrain', type=int, default=1, choices=[0,1], help='use mix training')
parser.add_argument('--mixtrain', type=int, help='use mix training')
parser.add_argument('--labelloss_weight', type=float, default=0.1, help='weight of h36m loss')

# teacher
parser.add_argument('--use_meanteacher', type=int, default=1,choices=[0,1], help='1: use mean teacher')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha * teacher + (1-alpha) * model, scale: 0-1')
parser.add_argument('--teacherloss_weight', type=float, default=0.1)

# motion
parser.add_argument('--use_motion', type=int, default=1,choices=[0,1], help='1: use mean teacher')
parser.add_argument('--interval', type=int, default=5, help='interval of temporal loss, scale: >= 1')
parser.add_argument('--motionloss_weight', type=float, default=0.8)


import h5py
class SourceDataset(Dataset):
    def __init__(self):
        super(SourceDataset, self).__init__()
        self.datainfos = {}
        fin = h5py.File('/dev/shm/h36m_part.h5', 'r')
        for k,v in fin.items():
            self.datainfos[k] = np.array(v)
        self.keys = self.datainfos.keys()
        print('finish loading data')
    
    def __getitem__(self, index):
        item = {}
        for k in self.datainfos.keys():
            item[k] = torch.from_numpy(self.datainfos[k][index]).float()
        return item
        
    
    def __len__(self,):
        return self.datainfos['pose'].shape[0]

class Adaptor():
    def __init__(self, options):
        self.options = options
        self.exppath = osp.join(self.options.expdir, self.options.expname)
        os.makedirs(self.exppath+'/mesh', exist_ok=True)
        os.makedirs(self.exppath+'/image', exist_ok=True)
        os.makedirs(self.exppath+'/result', exist_ok=True)
        self.summary_writer = SummaryWriter(self.exppath)
        self.device = torch.device('cuda')
        # set seed
        self.seed_everything(self.options.seed)

        self.options.mixtrain = self.options.lower_level_mixtrain or self.options.upper_level_mixtrain

        if self.options.retrieval:
            # # load basemodel's feature
            self.load_h36_cluster_res()

        if self.options.retrieval:
            self.h36m_dataset = SourceDataset()

        # set model
        self.set_model_optim()
        if self.options.use_meanteacher:
            self.set_teacher()
        
        # set dataset
        self.set_dataloader()

        # set criterion
        self.set_criterionn()

        self.setup_smpl()

    def get_h36m_data(self, indice):
        item_i = self.h36m_dataset[indice]
        return {k:v.unsqueeze(0) for k,v in item_i.items()}

    def load_h36_cluster_res(self,):
        ########## 0.1
        self.h36m_cluster_res = joblib.load('data/retrieval_res/cluster_res_random_sample_center_10_10_potocol2.pt')
        self.centers = self.h36m_cluster_res['centers']
        self.centers = torch.from_numpy(self.centers).float().to(self.device)
        self.index = self.h36m_cluster_res['index']
        self.h36m_base_features = np.concatenate(joblib.load('data/retrieval_res/h36m_feats_random_sample_center_10_10.pt'), axis=0)

    def retrieval(self, feature):
        dists = 1 - F.cosine_similarity(feature, self.centers)
        pos_cluster = torch.argsort(dists)[0].item()
        indices = self.index[pos_cluster]
        pos_indices = random.sample(indices, self.options.sample_num)
        h36mdata_list = []
        for x in pos_indices:
            h36mdata_list.append(self.get_h36m_data(x))
        h36m_batch = h36mdata_list[0]
        if len(h36mdata_list) > 1:
            for h36m_dataitem in h36mdata_list[1:]:
                for k,v in h36m_dataitem.items():
                    h36m_batch[k] = torch.cat([h36m_batch[k], v], dim=0)
        h36m_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in h36m_batch.items()}
        return h36m_batch


    def feature_cos_distance(self, feat1, feat2, sim=1):
        """
        1: expect feat1 and feat2 to be similar
        -1: expect feat1 and feat2 to be dissimilar
        """
        assert sim in [1, -1], print('sim should be -1 or 1')
        loss = self.cosembeddingloss(feat1, feat2, sim*torch.ones(feat1.shape[0]).to(self.device))
        return loss


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
        print('---> seed has been set')
    

    def set_model_optim(self,):
        """
        setup model and optimizer
        """
        checkpoint = torch.load(self.options.model_file)
        model = hmr(config.SMPL_MEAN_PARAMS)
        if self.options.use_boa:
            self.model = l2l.algorithms.MAML(model, lr=self.options.fastlr, first_order=True).to(self.device)
            # checkpoint['model'] = {'module.'+k: v for k, v in checkpoint['model'].items()}
            self.model.load_state_dict(checkpoint['model'], strict=True)
        else:
            self.model = model.to(self.device)
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
            self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.lr, betas=(self.options.beta1, self.options.beta2))
        self.base_params = {}
        for name, param in self.model.named_parameters():
            self.base_params[name] = param.clone().detach()
        print('---> model and optimizer have been set')
    

    def set_dataloader(self,):
        dataset = PW3D(self.options)
        self.dataloader = DataLoader(dataset, batch_size=self.options.batch_size, shuffle=False, num_workers=8)

    def set_criterionn(self,):
        self.gmm_f = MaxMixturePrior(prior_folder='data/spin_data', num_gaussians=8, dtype=torch.float32).to(self.device)
        self.cosembeddingloss = nn.CosineEmbeddingLoss().to(self.device)

    def setup_smpl(self,):
        self.smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(self.device)
        self.smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(self.device)
        self.smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(self.device)
        self.joint_mapper_h36m = constants.H36M_TO_J14
        self.joint_mapper_gt = constants.J24_TO_J14
        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    def set_teacher(self,):
        model = hmr(config.SMPL_MEAN_PARAMS)
        checkpoint = torch.load(self.options.model_file)
        for param in model.parameters():
            param.detach_()
        self.teacher = model.to(self.device)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        self.teacher.load_state_dict(checkpoint['model'], strict=True)

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


    def get_hist(self,):
        infos = self.history[self.global_step - self.options.interval]
        return torch.from_numpy(infos['image']).to(self.device), torch.from_numpy(infos['s2d']).to(self.device)


    def save_hist(self, image, s2d):
        self.history[self.global_step] = {'image': image.detach().cpu().numpy(), 
                                          's2d': s2d.detach().cpu().numpy()}
    

    def decode_smpl_params(self, poses, beta, gender='neutral', pose2rot=False):
        if gender == 'neutral':
            smpl_out = self.smpl_neutral(betas=beta, body_pose=poses[:,1:], global_orient=poses[:,0].unsqueeze(1), pose2rot=pose2rot)
        elif gender == 'male':
            smpl_out = self.smpl_male(betas=beta, body_pose=poses[:,1:], global_orient=poses[:,0].unsqueeze(1), pose2rot=pose2rot)
        elif gender == 'female':
            smpl_out = self.smpl_female(betas=beta, body_pose=poses[:,1:], global_orient=poses[:,0].unsqueeze(1), pose2rot=pose2rot)
        return {'s3d': smpl_out.joints, 'vts': smpl_out.vertices}


    def update_teacher(self, teacher, model):
        """
        teacher = teacher * alpha + model * (1 - alpha)
        In general, I set alpha to be 0.1.
        """
        factor = self.options.alpha
        for param_t, param in zip(teacher.parameters(), model.parameters()):
            # param_t.data.mul_(factor).add_(1 - factor, param.data)
            param_t.data.mul_(factor).add_(param.data, alpha=1 - factor)


    def excute(self,):
        self.sims = []
        self.feat_sims = {}
        self.optim_step_record = []
        mpjpe_all, pampjpe_all = [], []
        pve_all = []
        self.mpjpe_statistics, self.pampjpe_statistics = [[] for i in range(len(self.dataloader))], [[] for i in range(len(self.dataloader))]
        self.mpjpe_all_lower, self.pampjpe_all_lower = [[] for i in range(self.options.inner_step)], [[] for i in range(self.options.inner_step)] 
        self.history = {}
        self.kp2dlosses_lower = []
        self.kp2dlosses_upper = {}
        self.load = False
        for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            self.global_step = step
            self.fit_losses = {}
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            # Step1: adaptation
            self.model.eval()
            mpjpe, pampjpe, pve = self.adaptation(batch)

            # Step2: inference
            self.fit_losses['metrics/mpjpe'] = mpjpe
            self.fit_losses['metrics/pampjpe'] = pampjpe
            self.fit_losses['metrics/pve'] = pve
            self.write_summaries(self.fit_losses)
            mpjpe_all.append(mpjpe)
            pampjpe_all.append(pampjpe)
            pve_all.append(pve)
            if (self.global_step+1) % 200 == 0:
                print(f'Step:{self.global_step}: MPJPE:{np.mean(mpjpe_all)}, PAMPJPE:{np.mean(pampjpe_all)}, PVE:{np.mean(pve_all)}')
            if self.load:
                self.load_ckpt()

        # logout the results
        print('--- Final ---')
        print(f'Step:{self.global_step}: MPJPE:{np.mean(mpjpe_all)}, PAMPJPE:{np.mean(pampjpe_all)}, PVE:{np.mean(pve_all)}')
        for i in range(self.options.inner_step):
            print(f'Lower-level  Step:{i} MPJPE:{np.mean(self.mpjpe_all_lower[i])}, PAMPJPE:{np.mean(self.pampjpe_all_lower[i])}')
        
        # save results
        joblib.dump({'kp2dloss': self.kp2dlosses_lower}, osp.join(self.exppath, 'lowerlevel_kp2dloss.pt'))
        joblib.dump({'kp2dloss': self.kp2dlosses_upper}, osp.join(self.exppath, 'upperlevel_kp2dloss.pt'))

        joblib.dump({'mpjpe':mpjpe_all, 'pampjpe':pampjpe_all, 'pve':pve_all}, osp.join(self.exppath, 'res.pt'))
        joblib.dump({'mpjpe':self.mpjpe_all_lower, 'pampjpe':self.pampjpe_all_lower}, osp.join(self.exppath, 'lower_res.pt'))
        joblib.dump({'mpjpe':self.mpjpe_statistics, 'pampjpe':self.pampjpe_statistics}, osp.join(self.exppath, 'steps_statistic_res.pt'))
        joblib.dump({'feat': self.feat_sims}, osp.join(self.exppath, 'feat_sims.pt'))
        joblib.dump({'step': self.optim_step_record}, osp.join(self.exppath, 'optim_step_record.pt'))
        with open(osp.join(self.exppath, 'res.txt'), 'w') as f:
            f.write(f'Step:{self.global_step}: MPJPE:{np.mean(mpjpe_all)}, PAMPJPE:{np.mean(pampjpe_all)}, PVE:{np.mean(pve_all)}\n')
            for i in range(self.options.inner_step):
                f.write(f'Lower-level  Step:{i} MPJPE:{np.mean(self.mpjpe_all_lower[i])}, PAMPJPE:{np.mean(self.pampjpe_all_lower[i])}\n')


    def adaptation(self, batch):
        image = batch['image']
        gt_keypoints_2d = batch['smpl_j2d']
        self.save_hist(image, gt_keypoints_2d)

        if self.options.use_boa:
            with torch.no_grad():
                _,_,_, init_features = self.model(image, need_feature=True)
            h36m_batch = None
            # step 1, clone model
            learner = self.model.clone()
            # step 2, lower probe
            for i in range(self.options.inner_step):
                lower_level_loss, _ = self.lower_level_adaptation(image, gt_keypoints_2d, h36m_batch, learner)
                learner.adapt(lower_level_loss)
                # to evaluate the lower-level model
                mpjpe, pampjpe,_ = self.inference(batch, learner)
                self.fit_losses[f'metrics/lower_{i}_mpjpe'] = mpjpe
                self.fit_losses[f'metrics/lower_{i}_pampjpe'] = pampjpe
                self.mpjpe_all_lower[i].append(mpjpe)
                self.pampjpe_all_lower[i].append(pampjpe)
            # step 3, upper update
            upper_level_loss, _ = self.upper_level_adaptation(image, gt_keypoints_2d, h36m_batch, learner)
            self.optimizer.zero_grad()
            upper_level_loss.backward()
            self.optimizer.step()
            if self.options.use_meanteacher:
                # update the mean teacher
                self.update_teacher(self.teacher, self.model)

            # record pamjpe and mpjpe
            mpjpe, pampjpe, pve = self.inference(batch, self.model)
            self.mpjpe_statistics[self.global_step] = [mpjpe,]
            self.pampjpe_statistics[self.global_step] = [pampjpe,]

            if self.options.dynamic_boa:
                # cal similarity to judge whether this sample needs more optimization
                with torch.no_grad():
                    _,_,_, adapted_features = self.model(image, need_feature=True)
                    feat_sim_dict = self.cal_feature_diff(init_features, adapted_features)
                    feat_12 = feat_sim_dict[12]['cos']
                    self.feat_sims[self.global_step] = [feat_sim_dict]

                # while the 1-feat_12 not converge, continual optimizing.
                self.optimized_step = 0
                while 1-feat_12 > self.options.cos_sim_threshold:
                    self.optimized_step += 1
                    if self.optimized_step > self.options.optim_steps:
                        # maximun optimzation step, stop the optimization.
                        break
                    upper_level_loss, adapted_features = self.upper_level_adaptation(image, gt_keypoints_2d, h36m_batch, self.model)
                    self.optimizer.zero_grad()
                    upper_level_loss.backward()
                    self.optimizer.step()
                    if self.options.use_meanteacher:
                        self.update_teacher(self.teacher, self.model)
                    with torch.no_grad():
                        init_features = adapted_features
                        _,_,_, adapted_features = self.model(image, need_feature=True)
                        feat_sim_dict = self.cal_feature_diff(init_features, adapted_features)
                        feat_12 = feat_sim_dict[12]['cos']
                        self.feat_sims[self.global_step].append(feat_sim_dict)
                    # record pamjpe and mpjpe
                    mpjpe, pampjpe, pve = self.inference(batch, self.model)
                    self.mpjpe_statistics[self.global_step].append(mpjpe)
                    self.pampjpe_statistics[self.global_step].append(pampjpe)
                self.optim_step_record.append(self.optimized_step)
            return mpjpe, pampjpe, pve
        else:
            h36m_batch=None
            lower_level_loss, _ = self.lower_level_adaptation(image, gt_keypoints_2d, h36m_batch, self.model)
            self.optimizer.zero_grad()
            lower_level_loss.backward()
            self.optimizer.step()
            mpjpe, pampjpe, pve = self.inference(batch, self.model)
            return mpjpe, pampjpe, pve

    def cal_feature_diff(self, features_i, features_j):
        sims_dict = {}
        mean_cos_sim = 0
        for i, (feat_i, feat_j) in enumerate(zip(features_i, features_j)):
            cos_sim = F.cosine_similarity(feat_i.flatten(), feat_j.flatten(), dim=0, eps=1e-12)
            mean_cos_sim += cos_sim
            sims_dict[i] = {'cos': cos_sim.item(),}
        self.fit_losses['feat_sim/cos_sim'] = mean_cos_sim / i
        return sims_dict


    def lower_level_adaptation(self, image, gt_keypoints_2d, h36m_batch, learner=None):
        batch_size = image.shape[0]
        pred_rotmat, pred_shape, pred_cam, init_features = learner(image, need_feature=True)
        smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
        pred_s3d = smpl_out['s3d']
        pred_vertices = smpl_out['vts']
        pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
        conf = gt_keypoints_2d[:, 25:, -1].unsqueeze(-1).clone()
        
        if self.options.use_frame_losses_lower:
            # calculate losses
            # 2D keypoint loss
            s2dloss = (F.mse_loss(pred_s2d[:, 25:], gt_keypoints_2d[:, 25:, :-1], reduction='none')*conf).mean()
            # shape prior constraint
            shape_prior = self.cal_shape_prior(pred_shape)
            # pose prior constraint
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)
            loss = s2dloss * self.options.s2dloss_weight +\
                            shape_prior * self.options.shape_prior_weight +\
                            pose_prior * self.options.pose_prior_weight
            
            self.kp2dlosses_lower.append(s2dloss.item())
            self.fit_losses['ll/s2dloss'] = s2dloss
            self.fit_losses['ll/shape_prior'] = shape_prior
            self.fit_losses['ll/pose_prior'] = pose_prior
            self.fit_losses['ll/unlabelloss'] = loss

        if self.options.use_temporal_losses_lower:
            if self.options.use_meanteacher:
                teacherloss = self.cal_teacher_loss(image, pred_rotmat, pred_shape, pred_s2d, pred_s3d)
                if self.options.use_frame_losses_lower:
                    loss += teacherloss * self.options.teacherloss_weight
                else:
                    loss = teacherloss * self.options.teacherloss_weight

            if self.options.use_motion and (self.global_step - self.options.interval) > 0:
                motionloss = self.cal_motion_loss(learner, pred_s2d[:, 25:], gt_keypoints_2d[:, 25:], prefix='ul')
                loss += motionloss * self.options.motionloss_weight

        if self.options.retrieval:
            h36m_batch = self.retrieval(init_features[5])
        
        if self.options.lower_level_mixtrain:
            lableloss, label_feats = self.adapt_on_labeled_data(learner, h36m_batch, prefix='ll')
            loss += lableloss * self.options.labelloss_weight

        return loss, init_features


    def upper_level_adaptation(self, image, gt_keypoints_2d, h36m_batch, learner=None):
        batch_size = image.shape[0]
        pred_rotmat, pred_shape, pred_cam, init_features = learner(image, need_feature=True)
        smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
        pred_s3d = smpl_out['s3d']
        pred_vertices = smpl_out['vts']
        pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
        conf = gt_keypoints_2d[:, 25:, -1].unsqueeze(-1).clone()

        if self.options.use_frame_losses_upper:
            # calculate losses
            # 2D keypoint loss
            s2dloss = (F.mse_loss(pred_s2d[:, 25:], gt_keypoints_2d[:, 25:, :-1], reduction='none')*conf).mean()
            # shape prior constraint
            shape_prior = self.cal_shape_prior(pred_shape)
            # # pose prior constraint
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)
            loss = s2dloss * self.options.s2dloss_weight +\
                shape_prior * self.options.shape_prior_weight +\
                pose_prior * self.options.pose_prior_weight
            
            self.kp2dlosses_upper[self.global_step] = s2dloss.item()
            self.fit_losses['ul/s2dloss'] = s2dloss
            self.fit_losses['ul/shape_prior'] = shape_prior
            self.fit_losses['ul/pose_prior'] = pose_prior
            self.fit_losses['ul/unlabelloss'] = loss

        if self.options.use_temporal_losses_upper:
            if self.options.use_meanteacher:
                teacherloss = self.cal_teacher_loss(image, pred_rotmat, pred_shape, pred_s2d, pred_s3d)
                if self.options.use_frame_losses_upper:
                    loss += teacherloss * self.options.teacherloss_weight
                else:
                    loss = teacherloss * self.options.teacherloss_weight

            if self.options.use_motion and (self.global_step - self.options.interval) > 0:
                motionloss = self.cal_motion_loss(learner, pred_s2d[:, 25:], gt_keypoints_2d[:, 25:], prefix='ul')
                loss += motionloss * self.options.motionloss_weight

        if self.options.retrieval:
            h36m_batch = self.retrieval(init_features[5])

        if self.options.upper_level_mixtrain:
            lableloss, label_feats = self.adapt_on_labeled_data(learner, h36m_batch, prefix='ul')
            loss += lableloss * self.options.labelloss_weight

        return loss, init_features


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
        self.fit_losses['teacher/s2dloss'] = s2dloss
        self.fit_losses['teacher/s3dloss'] = s3dloss
        self.fit_losses['teacher/shape_loss'] = shape_loss
        self.fit_losses['teacher/pose_loss'] = pose_loss
        self.fit_losses['teacher/loss'] = loss
        return loss


    def adapt_on_labeled_data(self, model, batch, prefix='ll'):
        image = batch['img']
        gt_s3d = batch['pose_3d']
        gt_shape = batch['betas']
        gt_pose = batch['pose']
        gt_s2d = batch['keypoints']
        conf = gt_s2d[:, 25:, -1].unsqueeze(-1).clone()

        pred_rotmat, pred_shape, pred_cam, label_feats = model(image, need_feature=True)
        smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
        pred_s3d = smpl_out['s3d']
        pred_vertices = smpl_out['vts']

        # shape and pose losses
        gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)
        pose_loss = F.mse_loss(pred_rotmat, gt_rotmat)
        shape_loss = F.mse_loss(pred_shape, gt_shape)
        # 2d kp loss
        pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
        s2dloss = (F.mse_loss(pred_s2d[:, 25:],gt_s2d[:,25:,:-1], reduction='none') * conf).mean()
        # 3d kp loss
        s3dloss = self.cal_s3d_loss(pred_s3d[:, 25:], gt_s3d[:,:,:-1], conf)
        assert gt_s3d.shape[1] == 24
        
        loss = s2dloss * 5 + s3dloss * 5 + shape_loss * 0.001 + pose_loss * 1
        self.fit_losses[f'{prefix}/labled_s2dloss'] = s2dloss
        self.fit_losses[f'{prefix}/labled_s3dloss'] = s3dloss
        self.fit_losses[f'{prefix}/labled_shape_loss'] = shape_loss
        self.fit_losses[f'{prefix}/labled_pose_loss'] = pose_loss
        self.fit_losses[f'{prefix}/labled_loss'] = loss
        return loss, label_feats


    def cal_motion_loss(self, model, pred_s2d, gt_s2d, prefix='ul'):
        hist_image, hist_s2d = self.get_hist()
        hist_pred_rotmat, hist_pred_shape, hist_pred_cam = model(hist_image)
        hist_smpl_out = self.decode_smpl_params(hist_pred_rotmat, hist_pred_shape)
        hist_pred_s3d = hist_smpl_out['s3d']

        # cal motion loss
        hist_pred_s2d = self.projection(hist_pred_cam, hist_pred_s3d)['normed']
        pred_motion = pred_s2d - hist_pred_s2d[:,25:]
        gt_motion = gt_s2d[:,:,:-1] - hist_s2d[:,25:,:-1]
        # cal non-zero confidence
        conf1 = hist_s2d[:,25:, -1].unsqueeze(-1).clone()
        conf2 = gt_s2d[:,:, -1].unsqueeze(-1).clone()
        one = torch.tensor([1.]).to(self.device)
        zero = torch.tensor([0.]).to(self.device)
        conf = torch.where((conf1 + conf2)==2,one,zero)
        
        motion_loss = (F.mse_loss(pred_motion, gt_motion, reduction='none')*conf).mean()
        self.fit_losses[f'{prefix}/motion_loss'] = motion_loss
        return motion_loss


    def cal_shape_prior(self, pred_betas):
        return (pred_betas**2).sum(dim=-1).mean()
    

    def cal_pose_prior(self, pred_rotmat, betas):
        # gmm prior
        body_pose = rotation_matrix_to_angle_axis(pred_rotmat[:,1:].contiguous().view(-1,3,3)).contiguous().view(-1, 69)
        pose_prior_loss = self.gmm_f(body_pose, betas).mean()
        return pose_prior_loss


    def cal_s3d_loss(self, pred_s3d, gt_s3d, conf):
        """ 
        align the s3d and then cal the mse loss
        Input: (N,24,2)
        """
        gt_hip = (gt_s3d[:,2] + gt_s3d[:,3]) / 2
        gt_s3d = gt_s3d - gt_hip[:,None,:]
        pred_hip = (pred_s3d[:,2] + pred_s3d[:,3]) / 2
        pred_s3d = pred_s3d - pred_hip[:,None,:]
        loss = (conf * F.mse_loss(pred_s3d, gt_s3d, reduction='none')).mean()
        return loss


    def inference(self, batch, model, need_feature=False):
        image = batch['image']
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gender = batch['gender']
        
        model.eval()
        with torch.no_grad():
            if need_feature:
                pred_rotmat, pred_shape, pred_cam, features = model(image, need_feature)
            else:
                pred_rotmat, pred_shape, pred_cam = model(image, need_feature)
        smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
        pred_joints = smpl_out['s3d']
        pred_vertices = smpl_out['vts']

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(self.device)
        gt_vertices = self.smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
        gt_vertices_female = self.smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
        gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
        gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
        gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_h36m, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, self.joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        # 1. MPJPE
        mpjpe = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        # 2. PA-MPJPE
        S1 = pred_keypoints_3d.cpu().numpy()
        S2 = gt_keypoints_3d.cpu().numpy()
        S1_hat = compute_similarity_transform_batch(S1, S2)
        pampjpe = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
        # compute PVE
        smpl_out = self.smpl_neutral(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], pose2rot=True)
        gt_vertices = smpl_out.vertices.detach().cpu().numpy()
        pve = np.sqrt(np.sum((gt_vertices - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)).mean()

        # to cache results
        pred_cam_t = torch.stack([pred_cam[:,1],
                                    pred_cam[:,2],
                                    2* 5000./(constants.IMG_RES * pred_cam[:,0] +1e-9)],dim=-1)
        cached_results = {'verts': pred_vertices.cpu().numpy(),
                          'cam': pred_cam_t.cpu().numpy(),
                          'rotmat': pred_rotmat.cpu().numpy(),
                          'beta': pred_shape.cpu().numpy()}
        joblib.dump(cached_results, osp.join(self.exppath, 'result', f'Pred_{self.global_step}.pt'))

        # visulize results
        if self.options.save_res:
            self.save_results(pred_vertices, pred_cam, image, batch['imgname'], batch['bbox'],mpjpe*1000, pampjpe*1000, prefix='Pred')
        if need_feature:
            return mpjpe*1000, pampjpe*1000, pve*1000, features
        else:
            return mpjpe*1000, pampjpe*1000, pve*1000


    def save_results(self, vts, cam_trans, images, name, bbox, mpjpe, pampjpe, prefix=None):
        vts = vts.clone().detach().cpu().numpy()
        cam_trans = cam_trans.clone().detach().cpu().numpy()
        images = images.clone().detach()
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = np.transpose(images.cpu().numpy(), (0,2,3,1))
        for i in range(vts.shape[0]):
            oriimg = cv2.imread(os.path.join('/data/syguan/human_datasets/3dpw', name[i]))
            ori_h, ori_w = oriimg.shape[:2]
            bbox = bbox.cpu().numpy()
            ori_pred_cams = convert_crop_cam_to_orig_img(cam_trans, bbox, ori_w, ori_h)
            renderer = Renderer(resolution=(ori_w, ori_h), orig_img=True, wireframe=False)
            rendered_image = renderer.render(oriimg, vts[i], ori_pred_cams[i], color=np.array([205,129,98])/255, mesh_filename='demo.obj')
            cv2.imwrite(osp.join(self.exppath, 'image', f'{prefix}_{self.global_step+i}.png'), rendered_image)

    def write_summaries(self, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.global_step)

if __name__ == '__main__':
    options = parser.parse_args()
    exppath = osp.join(options.expdir, options.expname)
    os.makedirs(exppath, exist_ok=False)
    argsDict = options.__dict__
    with open(f'{exppath}/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    adaptor = Adaptor(options)
    adaptor.excute()
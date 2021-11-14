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
import numpy as np
import os.path as osp
import learn2learn as l2l


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize

from utils.dataprocess import crop,transform, rot_aa
import config
import constants
from model import SMPL, hmr
from utils.smplify.prior import MaxMixturePrior
from utils.geometry import batch_rodrigues, perspective_projection, rotation_matrix_to_angle_axis
from boa_dataset.pw3d import PW3D
from boa_dataset.internet_data import Internet_dataset
from render_demo import Renderer, convert_crop_cam_to_orig_img


class BaseAdaptor():
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
            self.h36m_dataset = SourceDataset(datapath='data/retrieval_res/h36m_random_sample_center_10_10.pt')

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
        return {k:v for k,v in item_i.items()}

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
        if self.options.dataset == '3dpw':
            dataset = PW3D(self.options)
            self.imgdir = config.PW3D_ROOT
        else:
            dataset = Internet_dataset()
            self.imgdir = osp.join(config.InternetData_ROOT, 'images')
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
        pass


    def adaptation(self):
        pass

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
        pass


    def save_results(self, vts, cam_trans, images, name, bbox, prefix=None):
        vts = vts.clone().detach().cpu().numpy()
        cam_trans = cam_trans.clone().detach().cpu().numpy()
        images = images.clone().detach()
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = np.transpose(images.cpu().numpy(), (0,2,3,1))
        for i in range(vts.shape[0]):
            oriimg = cv2.imread(os.path.join(self.imgdir, name[i]))
            ori_h, ori_w = oriimg.shape[:2]
            bbox = bbox.cpu().numpy()
            ori_pred_cams = convert_crop_cam_to_orig_img(cam_trans, bbox, ori_w, ori_h)
            renderer = Renderer(resolution=(ori_w, ori_h), orig_img=True, wireframe=False)
            rendered_image = renderer.render(oriimg, vts[i], ori_pred_cams[i], color=np.array([205,129,98])/255, mesh_filename='demo.obj')
            cv2.imwrite(osp.join(self.exppath, 'image', f'{prefix}_{self.global_step+i}.png'), rendered_image)

    def write_summaries(self, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.global_step)


class SourceDataset(Dataset):
    def __init__(self,datapath):
        super(SourceDataset, self).__init__()
        self.img_dir = config.H36M_ROOT
        self.data = joblib.load(datapath)
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        # == parse data == #
        self.imgname = self.data['imgname']
        # import ipdb;ipdb.set_trace()
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.pose = self.data['pose'].astype(np.float)
        self.betas = self.data['shape'].astype(np.float)
        self.pose_3d = self.data['S']
        keypoints_gt = self.data['part']
        keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)
        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        self.length = self.scale.shape[0]
    
    def __getitem__(self,index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        pose = self.pose[index].copy()
        betas = self.betas[index].copy()

        # Load image
        imgname = os.path.join(self.img_dir, self.imgname[index])
        img = self.read_image(imgname)
        item['oriimage'] = img.copy()
        orig_shape = np.array(img.shape)[:2]

        # no augmentation
        rot, sc = 0, 1

        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot)).float().unsqueeze(0)
        img = self.rgb_processing(img, center, sc*scale, rot)
        item['oriimage2'] = [img.copy(), center, sc*scale, rot]
        
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)

        item['img'] = img.unsqueeze(0)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot)).float().unsqueeze(0)
        item['betas'] = torch.from_numpy(betas).float().unsqueeze(0)
        item['imgname'] = imgname
        S = self.pose_3d[index].copy()
        item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot)).float().unsqueeze(0)
        return item

    def __len__(self):
        return len(self.imgname)

    def j2d_processing(self, kp, center, scale, r):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp
        
    def read_image(self, imgname):
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        return img

    def rgb_processing(self, rgb_img, center, scale, rot):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def pose_processing(self, pose, r):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # (72),float
        pose = pose.astype('float32')
        return pose
    
    def j3d_processing(self, S, r):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        S = S.astype('float32')
        return S


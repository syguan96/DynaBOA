"""
dynaboa
"""

import os
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch
import constants
from utils.pose_utils import compute_similarity_transform_batch
from base_adaptor import BaseAdaptor

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, default='exps')
parser.add_argument('--expname', type=str, default='3dpw')
parser.add_argument('--dataset', type=str, default='3dpw', choices=['3dpw', 'internet'])
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
parser.add_argument('--cos_sim_threshold', type=float, default=3.1e-4, help='cos sim threshold')
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



class Adaptor(BaseAdaptor):

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

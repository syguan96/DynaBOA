"""
dynaboa
"""

import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import joblib
import argparse
from tqdm import tqdm
import os.path as osp
import torch

import constants
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
        self.history = {} 
        self.kp2dlosses_lower = []
        self.kp2dlosses_upper = {}
        for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc='Adapt'):
            self.global_step = step
            self.fit_losses = {}
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            # Step1: adaptation
            self.model.eval()
            self.adaptation(batch)

            # Step2: inference
            self.inference(batch, self.model)


    def adaptation(self, batch):
        image = batch['image']
        gt_keypoints_2d = batch['smpl_j2d']
        self.save_hist(image, gt_keypoints_2d)

        # clone the intial model to calculate similarity
        with torch.no_grad():
             _,_,_, init_features = self.model(image, need_feature=True)

        h36m_batch = None

        # step 1, clone model
        learner = self.model.clone()
        # step 2, lower probe
        for i in range(self.options.inner_step):
            lower_level_loss, _ = self.lower_level_adaptation(image, gt_keypoints_2d, h36m_batch, learner)
            learner.adapt(lower_level_loss)
            
        # step 3, upper update
        upper_level_loss, _ = self.upper_level_adaptation(image, gt_keypoints_2d, h36m_batch, learner)
        self.optimizer.zero_grad()
        upper_level_loss.backward()
        self.optimizer.step()
        if self.options.use_meanteacher:
            self.update_teacher(self.teacher, self.model)

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


    def inference(self, batch, model, need_feature=False):
        image = batch['image']

        model.eval()
        with torch.no_grad():
            if need_feature:
                pred_rotmat, pred_shape, pred_cam, features = model(image, need_feature)
            else:
                pred_rotmat, pred_shape, pred_cam = model(image, need_feature)
        smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
        pred_vertices = smpl_out['vts']

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
            self.save_results(pred_vertices, pred_cam, image, batch['imgname'], batch['bbox'], prefix='Pred')


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

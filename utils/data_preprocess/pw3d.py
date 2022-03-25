import os
import cv2
import numpy as np
import pickle
import ipdb
from numpy.lib.function_base import append
from tqdm import tqdm
import torch

from utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
from model.smpl import SMPL
import config
from utils.kp_utils import get_perm_idxs


# --- predefined variables
# smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
scaleFactor = 1#1.2
device = torch.device('cpu')
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
# --- predefined variables end

def projection(smpl, smpl_trans, camPose, camIntrinsics):
    smpl += smpl_trans
    smpl = np.concatenate([smpl, np.ones((49, 1))], axis=1)
    smpl = np.dot(smpl, camPose.T)[:, :3]
    smpl /= smpl[:, np.newaxis, -1]
    smpl = np.dot(smpl, camIntrinsics.T)
    return smpl[:,:2]


def get_smpl_joints(gt_betas, gt_pose, gender):
    if len(gt_betas.shape) ==1 and len(gt_pose.shape) == 1:
        gt_betas = torch.from_numpy(gt_betas).float().unsqueeze(0)
        gt_pose = torch.from_numpy(gt_pose).float().unsqueeze(0)
    else:
        gt_betas = torch.from_numpy(gt_betas).float()
        gt_pose = torch.from_numpy(gt_pose).float()
    if gender == 'm':
        gt_joints = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints 
    elif gender == 'f':
        gt_joints = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints
    gt_joints = gt_joints.squeeze().numpy()
    return gt_joints


def get_bbox(j2d):
    bbox = [min(j2d[:,0]), min(j2d[:,1]),
                        max(j2d[:,0]), max(j2d[:,1])]
    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
    return center, scale


def pw3d_extract(dataset_path, out_path, debug=False):
    """
    only extract openpose detected dataset.
    """
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # annots we use
    imagenames_seq = []
    genders_seq = []
    scales_seq, centers_seq = [], []
    poses_seq, shapes_seq = [], []
    j3d_seq, j2d_seq, op_j2d_seq = [], [], []
 
    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    # dirty code, since we use it to keep the file order same.
    tmp = ['downtown_runForBus_00.pkl', 'downtown_rampAndStairs_00.pkl', 'flat_packBags_00.pkl', \
           'downtown_runForBus_01.pkl', 'office_phoneCall_00.pkl', 'downtown_windowShopping_00.pkl', \
           'downtown_walkUphill_00.pkl', 'downtown_sitOnStairs_00.pkl', 'downtown_enterShop_00.pkl', \
           'downtown_walking_00.pkl', 'downtown_stairs_00.pkl', 'downtown_crossStreets_00.pkl', \
           'downtown_car_00.pkl', 'downtown_downstairs_00.pkl', 'downtown_bar_00.pkl', \
           'downtown_walkBridge_01.pkl', 'downtown_weeklyMarket_00.pkl', 'downtown_warmWelcome_00.pkl', \
           'downtown_arguing_00.pkl', 'downtown_upstairs_00.pkl', 'downtown_bus_00.pkl', 'flat_guitar_01.pkl', 'downtown_cafe_00.pkl', 'outdoors_fencing_01.pkl']
    sequences = [os.path.join(dataset_path, f) for f in tmp]
    for seq_idx, seq in tqdm(enumerate(sequences), total=len(sequences)):
        with open(seq, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        assert (data['poses2d'][0].shape[0] == num_frames)
        for p_id in range(num_people):
            campose_valid = np.array(data['campose_valid']).astype(np.bool)
            campose_valid = campose_valid[p_id]
            poses = data['poses'][p_id][campose_valid]
            shapes = data['betas'][p_id][:10]
            shapes = np.tile(shapes.reshape(1, -1), (num_frames, 1))
            shapes = shapes[campose_valid]

            trans = data['trans'][p_id][campose_valid]
            openpose_j2d = data['poses2d'][p_id].transpose(0,2,1)
            openpose_j2d = openpose_j2d[campose_valid]
            cam_pose = data['cam_poses'][campose_valid]
            gender = data['genders'][p_id]
            trans = data['trans'][p_id][campose_valid]
            cam_Intrinsics = data['cam_intrinsics']
            seq_name = str(data['sequence'])
            imagenames = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            imagenames = imagenames[campose_valid]

            # collect 3D joints
            j3ds = get_smpl_joints(shapes, poses, gender)
            # project smpl's joints to 2D plane
            gt_j2ds = []
            for i in range(j3ds.shape[0]):
                gt_j2d= projection(j3ds[i], trans[i], cam_pose[i], cam_Intrinsics)
                gt_j2d = np.concatenate([gt_j2d, np.ones((49,1))], axis=1)
                gt_j2ds.append(gt_j2d)
            gt_j2ds = np.stack(gt_j2ds, 0)
            assert gt_j2ds.shape[1] == 49, print(gt_j2ds.shape)
            # collect openpose 2d keypoints
            j49s = np.zeros_like(gt_j2ds)
            # import ipdb;ipdb.set_trace()
            j49s[:,idx] = openpose_j2d

            # calculate bbox
            centers, scales = [], []
            for i in range(gt_j2ds.shape[0]):
                center_, scale_ = get_bbox(gt_j2ds[i])
                centers.append(center_)
                scales.append(scale_)
            centers = np.stack(centers, axis=0)
            scales = np.stack(scales, axis=0)

            # align the mesh rotation
            root_rot = torch.from_numpy(poses[:, :3]).float()
            root_rotmat = batch_rodrigues(root_rot)
            Rc = torch.from_numpy(cam_pose[:, :3, :3]).float()
            Rs = torch.bmm(Rc, root_rotmat.reshape(-1, 3, 3))
            root_rot = rotation_matrix_to_angle_axis(Rs)
            poses[:,:3] = root_rot.numpy()

            out_file = os.path.join(out_path,f'3dpw_{seq_idx}_{p_id}.npz')
            np.savez(out_file, imgname=imagenames,
                                gender=np.array([gender]*poses.shape[0]),
                                scale=scales,
                                center=centers,
                                pose=poses,
                                shape=shapes,
                                j3d=j3ds,
                                j2d=gt_j2ds,
                                op_j2d=j49s)

            imagenames_seq.append(imagenames)
            genders_seq.append([gender]*poses.shape[0])
            scales_seq.append(scales)
            centers_seq.append(centers)
            poses_seq.append(poses)
            shapes_seq.append(shapes)
            j3d_seq.append(j3ds)
            j2d_seq.append(gt_j2ds)
            op_j2d_seq.append(j49s)

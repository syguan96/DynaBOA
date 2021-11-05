"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def compute_pck(s1, s2, threshold):
    error_pck = []
    for kp1, kp2 in zip(s1, s2):
        kp_diff_pa = np.linalg.norm(kp1 - kp2, axis=1)
        pa_pck = np.mean(kp_diff_pa < threshold)
        error_pck.append(pa_pck)
        error_pck = np.stack(error_pck)
    return error_pck


def reconstruction_error(S1, S2, alpha=0.15, needpck=False, needauc=False, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
   
    
    if needpck:
        pck_150 = compute_pck(S1, S2, threshold=0.15)
    
    if needauc:
        """
        refer to PoseAug
        """
        thresholds = np.linspace(0,150,31)
        pck_list = []
        for threshold in thresholds:
            _pck = compute_pck(S1, S2, threshold/1000)
            pck_list.append(_pck)
        auc =np.mean(pck_list)
    # error_pck = []
    # for kp1, kp2 in zip(S1_hat, S2):
    #     kp_diff_pa = np.linalg.norm(kp1 - kp2, axis=1)
    #     pa_pck = np.mean(kp_diff_pa < alpha)
    #     error_pck.append(pa_pck)
    #     error_pck = np.stack(error_pck)

    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)

    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    if needauc and needpck:
        return re, pck_150, auc
    elif needauc:
        return re, auc
    elif needpck:
        return re, pck_150
    else:
        return re

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

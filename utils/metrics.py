import sys
sys.path.append('..')

import torch
import numpy as np
import pickle

from utils.misc import get_rot_matrix, inverse_RT
from utils.genpose_utils import get_pose_dim
from ipdb import set_trace

def rot_diff_rad(rot1, rot2, chosen_axis=None, flip_axis=False):
    if chosen_axis is not None:
        axis = {'x': 0, 'y': 1, 'z': 2}
        y1, y2 = rot1[..., axis], rot2[..., axis]  # [Bs, 3]
        diff = torch.sum(y1 * y2, dim=-1)  # [Bs]
        diff = torch.clamp(diff, min=-1.0, max=1.0)
        rad = torch.acos(diff)
        if not flip_axis:
            return rad
        else:
            return torch.min(rad, np.pi - rad)

    else:
        mat_diff = torch.matmul(rot1, rot2.transpose(-1, -2))
        diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
        diff = (diff - 1) / 2.0
        diff = torch.clamp(diff, min=-1.0, max=1.0)
        return torch.acos(diff)


def rot_diff_degree(rot1, rot2, chosen_axis=None, flip_axis=False):
    return rot_diff_rad(rot1, rot2, chosen_axis=chosen_axis, flip_axis=flip_axis) / np.pi * 180.0


def get_trans_error(trans_1, trans_2):
    diff  = torch.norm(trans_1 - trans_2, dim=-1)
    return diff


def get_rot_error(rot_1, rot_2, error_mode, chosen_axis=None, flip_axis=False):
    assert error_mode in ['radian', 'degree'], f"the rotation error mode {error_mode} is not supported!"
    if error_mode == 'radian':
        rot_error = rot_diff_rad(rot_1, rot_2, chosen_axis, flip_axis)
    else:
        rot_error = rot_diff_degree(rot_1, rot_2, chosen_axis, flip_axis)
    return rot_error
    

def get_metrics_single_category(pose_1, pose_2, pose_mode, error_mode, chosen_axis=None, flip_axis=False, o2c_pose=False):
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"

    if pose_mode == 'rot_matrix':
        index = 6
    elif pose_mode == 'euler_xyz':
        index = 3
    else:
        index = 4

    rot_1 = pose_1[:, :index]
    rot_2 = pose_2[:, :index]
    trans_1 = pose_1[:, index:]
    trans_2 = pose_2[:, index:]
    
    rot_matrix_1 = get_rot_matrix(rot_1, pose_mode)
    rot_matrix_2 = get_rot_matrix(rot_2, pose_mode)
    
    if o2c_pose == False:
        rot_matrix_1, trans_1 = inverse_RT(rot_matrix_1, trans_1)
        rot_matrix_2, trans_2 = inverse_RT(rot_matrix_2, trans_2)
        
    rot_error = get_rot_error(rot_matrix_1, rot_matrix_2, error_mode, chosen_axis, flip_axis)
    trans_error = get_trans_error(trans_1, trans_2)
    
    return rot_error.cpu().numpy(), trans_error.cpu().numpy()


def compute_RT_errors(RT_1, RT_2, class_id, handle_visibility, synset_names):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]
    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]
    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
        (synset_names[class_id] == 'mug' and handle_visibility == 0):
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    else:
        R = R1 @ R2.transpose()
        cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

'''
def compute_RT_overlaps(gt_class_ids, gt_sRT, gt_handle_visibility, pred_class_ids, pred_sRT, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_errors(pred_sRT[i], gt_sRT[j], gt_class_ids[j],
                                                  gt_handle_visibility[j], synset_names)
    return overlaps

'''


def compute_RT_overlaps(class_ids, gt_RT, pred_RT, gt_handle_visibility, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:

    """
    num = len(class_ids)
    overlaps = np.zeros((num, 2))

    for i in range(num):
        overlaps[i, :] = compute_RT_errors(pred_RT[i], gt_RT[i], class_ids[i],
                                                gt_handle_visibility[i], synset_names)
    return overlaps


def get_metrics(pose_1, pose_2, class_ids, synset_names, gt_handle_visibility, pose_mode, o2c_pose=False):
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"

    index = get_pose_dim(pose_mode) - 3

    rot_1 = pose_1[:, :index]
    rot_2 = pose_2[:, :index]
    trans_1 = pose_1[:, index:]
    trans_2 = pose_2[:, index:]
    
    rot_matrix_1 = get_rot_matrix(rot_1, pose_mode)
    rot_matrix_2 = get_rot_matrix(rot_2, pose_mode)
    
    if o2c_pose == False:
        rot_matrix_1, trans_1 = inverse_RT(rot_matrix_1, trans_1)
        rot_matrix_2, trans_2 = inverse_RT(rot_matrix_2, trans_2)
    
    bs = pose_1.shape[0]
    RT_1 = torch.eye(4).unsqueeze(0).repeat([bs, 1, 1])
    RT_2 = torch.eye(4).unsqueeze(0).repeat([bs, 1, 1])
    
    RT_1[:, :3, :3] = rot_matrix_1
    RT_1[:, :3, 3] = trans_1
    RT_2[:, :3, :3] = rot_matrix_2
    RT_2[:, :3, 3] = trans_2
    
    error = compute_RT_overlaps(class_ids, RT_1.cpu().numpy(), RT_2.cpu().numpy(), gt_handle_visibility, synset_names)
    rot_error = error[:, 0]
    trans_error = error[:, 1]
    return rot_error, trans_error


if __name__ == '__main__':
    gt_pose = torch.rand(8, 7)
    gt_pose[:, :4] /= torch.norm(gt_pose[:, :4], dim=-1, keepdim=True)
    noise_pose = gt_pose + torch.rand(8, 7) / 10
    noise_pose[:, :4] /= torch.norm(noise_pose[:, :4], dim=-1, keepdim=True)
    rot_error = get_rot_error(gt_pose[:, :4], noise_pose[:, :4], 'camera', 'quat_wxyz', 'degree')
    trans_error = get_trans_error(gt_pose[:, 4:], noise_pose[:, 4:])
    print(rot_error, trans_error)



import torch
import torch.nn.functional as F
import numpy as np

from ipdb import set_trace


def get_pose_dim(rot_mode):
    assert rot_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'], \
        f"the rotation mode {rot_mode} is not supported!"
        
    if rot_mode == 'quat_wxyz' or rot_mode == 'quat_xyzw':
        pose_dim = 7
    elif rot_mode == 'euler_xyz':
        pose_dim = 6
    elif rot_mode == 'euler_xyz_sx_cx' or rot_mode == 'rot_matrix':
        pose_dim = 9
    else:
        raise NotImplementedError
    return pose_dim

'''
def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = F.normalize(x_raw, p=2, dim=-1)  # bx3
    z = torch.cross(x, y_raw, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    y = torch.cross(z, x, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)
'''

def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)  # b*3
    z = np.cross(x, y_raw) # b*3
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)          # b*3
    y = np.cross(z, x)     # b*3                      

    return np.stack((x, y, z), axis=-1)  # (b,3,3)


class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


def merge_results(results_ori, results_new):
    if len(results_ori.keys()) == 0:
        return results_new
    else:
        results = {
            'pred_pose': torch.cat([results_ori['pred_pose'], results_new['pred_pose']], dim=0),
            'gt_pose': torch.cat([results_ori['gt_pose'], results_new['gt_pose']], dim=0),
            'cls_id': torch.cat([results_ori['cls_id'], results_new['cls_id']], dim=0),
            'handle_visibility': torch.cat([results_ori['handle_visibility'], results_new['handle_visibility']], dim=0),
            # 'path': results_ori['path'] + results_new['path'],
        }
        return results



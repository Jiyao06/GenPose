import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from ipdb import set_trace
from utils.genpose_utils import get_pose_dim
from utils.metrics import get_metrics


class RewardModel(nn.Module):
    def __init__(self, pose_mode):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
        """
        super(RewardModel, self).__init__()
        pose_dim = get_pose_dim(pose_mode)
        self.act = nn.ReLU(True)
        
        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        ''' decoder '''
        self.reward_layer = nn.Sequential(
            nn.Linear(1024+256, 256),
            self.act,
            nn.Linear(256, 2),
        )

    def forward(
        self,
        pts_feature,
        pose
    ):
        """
        calculate the score of every pose

        Args:
            pts_feature (torch.tensor): [batch, 1024]
            pred_pose (torch.tensor): [batch, pose_dim]
        Returns:
            reward (torch.tensor): [batch, 2], the score of the pose estimation results, 
                the first item is rotation score and the second item is translation score.
        """
        
        pose_feature = self.pose_encoder(pose)
        feature = torch.cat((pts_feature, pose_feature), dim=-1)    # [bs, 1024+256]
        reward = self.reward_layer(feature)       # (batch, 1)
        return reward


def sort_results(energy, metrics):
    """ Sorting the results according to the pose error (low to high)

    Args:
        energy (torch.tensor): [bs, repeat_num, 2]
        metrics (torch.tensor): [bs, repeat_num, 2]
        
    Return:
        sorted_energy (torch.tensor): [bs, repeat_num, 2]
    """
    rot_error = metrics[..., 0]
    trans_error = metrics[..., 1]
    
    rot_index = torch.argsort(rot_error, dim=1, descending=False)
    trans_index = torch.argsort(trans_error, dim=1, descending=False)
    
    sorted_energy = energy.clone()
    sorted_energy[..., 0] = energy[..., 0].gather(1, rot_index)
    sorted_energy[..., 1] = energy[..., 1].gather(1, trans_index)
    
    return sorted_energy


# def ranking_loss(energy):
#     """ Calculate the ranking loss

#     Args:
#         energy (torch.tensor): [bs, repeat_num, 2]

#     Returns:
#         loss (torch.tensor)
#     """    
#     loss, count = 0, 0
#     repeat_num = energy.shape[1]
       
#     for i in range(repeat_num - 1):
#         for j in range(i+1, repeat_num):
#             # diff = torch.log(torch.sigmoid(score[:, i, :] - score[:, j, :]))
#             diff = torch.sigmoid(-energy[:, i, :] + energy[:, j, :])
#             loss += torch.mean(diff)
#             count += 1
#     loss = loss / count
#     return loss



def ranking_loss(energy):
    """ Calculate the ranking loss

    Args:
        energy (torch.tensor): [bs, repeat_num, 2]

    Returns:
        loss (torch.tensor)
    """    
    loss, count = 0, 0
    repeat_num = energy.shape[1]
       
    for i in range(repeat_num - 1):
        for j in range(i+1, repeat_num):
            # diff = torch.log(torch.sigmoid(score[:, i, :] - score[:, j, :]))
            diff = 1 + (-energy[:, i, :] + energy[:, j, :]) / (torch.abs(energy[:, i, :] - energy[:, j, :]) + 1e-5)
            loss += torch.mean(diff)
            count += 1
    loss = loss / count
    return loss


def sort_poses_by_energy(poses, energy):
    """  Rank the poses from highest to lowest energy 
    
    Args:
        poses (torch.tensor): [bs, inference_num, pose_dim]
        energy (torch.tensor): [bs, inference_num, 2]
        
    Returns:
        sorted_poses (torch.tensor): [bs, inference_num, pose_dim]
        sorted_energy (torch.tensor): [bs, inference_num, 2]        
    """
    # get the sorted energy 
    bs = poses.shape[0]
    repeat_num= poses.shape[1]
    sorted_energy, indices_1 = torch.sort(energy, descending=True, dim=1)
    indices_0 = torch.arange(0, energy.shape[0]).view(1, -1).to(energy.device).repeat(1, repeat_num)
    indices_1_rot = indices_1.permute(2, 1, 0)[0].reshape(1, -1)
    indices_1_trans = indices_1.permute(2, 1, 0)[1].reshape(1, -1)
    rot_index = torch.cat((indices_0, indices_1_rot), dim=0).cpu().numpy().tolist()
    trans_index = torch.cat((indices_0, indices_1_trans), dim=0).cpu().numpy().tolist()
    sorted_poses = poses[rot_index]
    sorted_poses[:, -3:] = poses[trans_index][:, -3:]
    sorted_poses = sorted_poses.view(repeat_num, bs, -1).permute(1, 0, 2)
    
    return sorted_poses, sorted_energy


def test_ranking_loss():
    energy = torch.tensor([[[100, 100],
                           [9, 9],
                           [8, 8],
                           [10, 10]]])
    loss = ranking_loss(energy)
    print(loss)
    
if __name__ == '__main__':
    test_ranking_loss()
    # bs = 3
    # repeat_num = 5
    # pts_feature = torch.randn(bs, 1024)
    # pred_pose = torch.randn(bs, repeat_num, 7)
    # metrics = torch.randn(bs, repeat_num, 2)
    
    # reward_model = RewardModel(pose_mode='quat_wxyz')
    # reward = reward_model(pts_feature.unsqueeze(1).repeat(1, repeat_num, 1), pred_pose)
    # sorted_reward = sort_results(reward, metrics)
    # loss = ranking_loss(sorted_reward)


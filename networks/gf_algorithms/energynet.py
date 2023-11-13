import copy
import torch
import torch.nn as nn

from ipdb import set_trace
from torch.autograd import Variable
from networks.gf_algorithms.samplers import cond_ode_likelihood, cond_ode_sampler, cond_pc_sampler
from networks.gf_algorithms.scorenet import GaussianFourierProjection
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from networks.pts_encoder.pointnets import PointNetfeat
from utils.genpose_utils import get_pose_dim


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


class PoseEnergyNet(nn.Module):
    def __init__(
        self, 
        marginal_prob_func, 
        pose_mode='quat_wxyz',
        regression_head='RT', 
        device='cuda', 
        energy_mode='L2', # ['DAE', 'L2', 'IP']
        s_theta_mode='score', # ['score', 'decoder', 'identical'])
        norm_energy='identical' # ['identical', 'std', 'minus']
    ):
        super(PoseEnergyNet, self).__init__()
        self.device = device
        self.regression_head = regression_head
        self.act = nn.ReLU(True)
        self.pose_dim = get_pose_dim(pose_mode)
        self.energy_mode = energy_mode
        self.s_theta_mode = s_theta_mode
        self.norm_energy = norm_energy

        ''' encode pose '''
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
        )
        
        ''' encode t '''
        self.t_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=128),
            # self.act,
            nn.Linear(128, 128),
            self.act,
        )

        ''' fusion tail '''
        if self.regression_head == 'RT':
            self.fusion_tail = nn.Sequential(
                nn.Linear(128+256+1024, 512),
                self.act,
                zero_module(nn.Linear(512, self.pose_dim)),
            )


        elif self.regression_head == 'R_and_T':
            ''' rotation regress head '''
            self.fusion_tail_rot = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                zero_module(nn.Linear(256, self.pose_dim-3)),
            )
            
            ''' tranalation regress head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                zero_module(nn.Linear(256, 3)),
            )
            
        
        elif self.regression_head == 'Rx_Ry_and_T':
            if pose_mode != 'rot_matrix':
                raise NotImplementedError
            ''' rotation_x_axis regress head '''
            self.fusion_tail_rot_x = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                zero_module(nn.Linear(256, 3)),
            )
            self.fusion_tail_rot_y = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                zero_module(nn.Linear(256, 3)),
            )
            
            ''' tranalation regress head '''
            self.fusion_tail_trans = nn.Sequential(
                nn.Linear(128+256+1024, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                self.act,
                zero_module(nn.Linear(256, 3)),
            )            

        else:
            raise NotImplementedError
        
        
        self.marginal_prob_func = marginal_prob_func
        
        
    def output_zero_initial(self):
        if self.regression_head == 'RT':
            zero_module(self.fusion_tail[-1])

        elif self.regression_head == 'R_and_T':
            zero_module(self.fusion_tail_rot[-1])
            zero_module(self.fusion_tail_trans[-1])
                 
        elif self.regression_head == 'Rx_Ry_and_T':
            zero_module(self.fusion_tail_rot_x[-1])
            zero_module(self.fusion_tail_rot_y[-1])
            zero_module(self.fusion_tail_trans[-1])  
        else:
            raise NotImplementedError
        

    def get_energy(self, pts_feat, sampled_pose, t, decoupled_rt=True):
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)

        total_feat = torch.cat([pts_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        
        ''' get f_{theta} '''
        if self.regression_head == 'RT':
            f_theta = self.fusion_tail(total_feat)
        elif self.regression_head == 'R_and_T':
            rot = self.fusion_tail_rot(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            f_theta = torch.cat([rot, trans], dim=-1)
        elif self.regression_head == 'Rx_Ry_and_T':
            rot_x = self.fusion_tail_rot_x(total_feat)
            rot_y = self.fusion_tail_rot_y(total_feat)
            trans = self.fusion_tail_trans(total_feat)
            f_theta = torch.cat([rot_x, rot_y, trans], dim=-1)
        else:
            raise NotImplementedError
        
        ''' get s_{theta} '''
        if self.s_theta_mode == 'score': 
            s_theta = f_theta / std
        elif self.s_theta_mode == 'decoder':
            s_theta = sampled_pose - std * f_theta
        elif self.s_theta_mode == 'identical':
            s_theta = f_theta
        else:
            raise NotImplementedError
        
        ''' get energy '''
        if self.energy_mode == 'DAE':
            energy = - 0.5 * torch.sum((sampled_pose - s_theta) ** 2, dim=-1)
        elif self.energy_mode == 'L2':
            energy = - 0.5 * torch.sum(s_theta ** 2, dim=-1)
        elif self.energy_mode == 'IP': # Inner Product
            energy = torch.sum(sampled_pose * s_theta, dim=-1)
            if decoupled_rt:
                energy_rot = torch.sum(sampled_pose[:, :-3] * s_theta[:, :-3], dim=-1)
                energy_trans = torch.sum(sampled_pose[:, -3:] * s_theta[:, -3:], dim=-1)
                energy = torch.cat((energy_rot.unsqueeze(-1), energy_trans.unsqueeze(-1)), dim=-1)
        else:
            raise NotImplementedError
        
        ''' normalisation '''
        if self.norm_energy == 'identical':
            pass
        elif self.norm_energy == 'std':
            energy = energy / (std + 1e-7)
        elif self.norm_energy == 'minus': # Inner Product
            energy = - energy
        else:
            raise NotImplementedError
        return energy


    def forward(self, data, return_item='score'):
        pts_feat = data['pts_feat']
        sampled_pose = data['sampled_pose']
        t = data['t']

        if return_item == 'energy':
            energy = self.get_energy(pts_feat, sampled_pose, t)
            return energy
        
        with TemporaryGrad():
            inp_variable_sampled_pose = Variable(sampled_pose, requires_grad=True)
            energy = self.get_energy(pts_feat, inp_variable_sampled_pose, t, decoupled_rt=False)
            scores, = torch.autograd.grad(energy, inp_variable_sampled_pose,
                                    grad_outputs=energy.data.new(energy.shape).fill_(1),
                                    create_graph=True)
        # inp_variable_sampled_pose = None # release the variable 
        if return_item == 'score':
            return scores
        elif return_item == 'score_and_energy':
            return scores, energy
        else:
            raise NotImplementedError



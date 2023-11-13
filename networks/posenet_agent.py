import sys
import os
import torch
import time
import functools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch3d
import random

from tensorboardX import SummaryWriter
from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from networks.gf_algorithms.score_utils import ExponentialMovingAverage
from networks.gf_algorithms.losses import loss_fn, loss_fn_edm
from networks.gf_algorithms.sde import init_sde
from networks.posenet import GFObjectPose

# from networks.gf_algorithms.sde_backup import ExponentialMovingAverage, loss_fn, loss_fn_edm, init_sde
# from networks.gf_algorithms.energynet import GFObjectPose
from networks.reward import sort_results, ranking_loss, sort_poses_by_energy
from utils.genpose_utils import TrainClock
from utils.misc import exists_or_mkdir, average_quaternion_batch
from utils.visualize import create_grid_image, test_time_visulize
from utils.metrics import get_metrics, get_rot_matrix


 
def get_ckpt_and_writer_path(cfg):
        if cfg.use_pretrain and not os.path.exists(f"./results/ckpts/{cfg.log_dir}/ckpt_epoch{cfg.model_name}.pth"):
            raise Exception(f"./results/ckpts/{cfg.log_dir}/ckpt_epoch{cfg.model_name}.pth is not exist!")
        
        ''' init exp folder and writer '''
        ckpt_path = f'./results/ckpts/{cfg.log_dir}'
        writer_path = f'./results/logs/{cfg.log_dir}' if cfg.use_pretrain == False else f'./results/logs/{cfg.log_dir}_continue'
        
        if cfg.is_train:
            exists_or_mkdir('./results')
            exists_or_mkdir(ckpt_path)
            exists_or_mkdir(writer_path)    
        return ckpt_path, writer_path    


class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        
        self.cfg = cfg
        self.is_testing = False
        self.clock = TrainClock()
        self.pts_feature = False

        # get checkpoint and writer path
        self.model_dir, writer_path = get_ckpt_and_writer_path(self.cfg)
        
        # init writer
        if self.cfg.is_train:
            self.writer = SummaryWriter(writer_path)
        
        # init sde
        self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T = init_sde(self.cfg.sde_mode)
        self.net = self.build_net()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=self.cfg.ema_rate)

        # init related functions
        if self.cfg.sde_mode == 'edm':
            self.loss_fn = functools.partial(loss_fn_edm, sigma_data=1.4148, P_mean=-1.2, P_std=1.2)
        else:
            self.loss_fn = loss_fn
         

    def get_network(self, name):
        if name == 'GFObjectPose':
            return GFObjectPose(self.cfg, self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps, self.T)
        else:
            raise NotImplementedError(f"Got name '{name}'")
    
    
    def build_net(self):
        net = self.get_network('GFObjectPose')
        net = net.to(self.cfg.device)
        if self.cfg.parallel:
            device_ids = list(range(self.cfg.num_gpu))
            net = nn.DataParallel(net, device_ids=device_ids).cuda()
        return net
    

    def set_optimizer(self):
        """set optimizer used in training"""
        params = []
        params = self.net.parameters()            
        self.base_lr = self.cfg.lr
        if self.cfg.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.cfg.optimizer == 'Adam':
            optimizer = optim.Adam(params, betas=(0.9, 0.999), eps=1e-8, lr=self.cfg.lr)     
        else:
            raise NotImplementedError
        return optimizer


    def set_scheduler(self):
        """set lr scheduler used in training"""
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay)
        return scheduler


    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.to(self.cfg.device)
        self.ema.restore(self.net.parameters())
            

    def load_ckpt(self, name=None, model_dir=None, model_path=False, load_model_only=False):
        """load checkpoint from saved checkpoint"""
        if not model_path:
            if name == 'latest':
                pass
            elif name == 'best':
                pass
            else:
                name = "ckpt_epoch{}".format(name)

            if model_dir is None:
                load_path = os.path.join(self.model_dir, "{}.pth".format(name))
            else:
                load_path = os.path.join(model_dir, "{}.pth".format(name))
        else:
            load_path = model_dir
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        
        if not load_model_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.clock.restore_checkpoint(checkpoint['clock'])


    def collect_score_loss(self, data, teacher_model=None, pts_feat_teacher=None):
        '''
        Args:
            data, dict {
                'pts': [bs, c]
                'gt_pose': [bs, pose_dim]
            }
        '''
        gf_loss = 0
        for _ in range(self.cfg.repeat_num):
            gf_loss += self.loss_fn(
                model=self.net,
                data=data,
                marginal_prob_func=self.marginal_prob_fn,
                sde_fn=self.sde_fn,
                likelihood_weighting=self.cfg.likelihood_weighting, 
                teacher_model=teacher_model,
                pts_feat_teacher=pts_feat_teacher
            )
        gf_loss /= self.cfg.repeat_num
        losses = {'gf': gf_loss}        
        return losses


    def collect_ema_loss(self, data):
        '''
        Args:
            data, dict {
                'pts': [bs, c]
                'gt_pose': [bs, pose_dim]
            }
        '''
        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        with torch.no_grad():
            ema_loss = 0
            for _ in range(self.cfg.repeat_num):
                # calc score-matching loss
                ema_loss += self.loss_fn(
                    model=self.net,
                    data=data,
                    marginal_prob_func=self.marginal_prob_fn,
                    sde_fn=self.sde_fn,
                    likelihood_weighting=self.cfg.likelihood_weighting
                )
            ema_loss /= self.cfg.repeat_num
        self.ema.restore(self.net.parameters())
        ema_losses = {'ema': ema_loss}        
        return ema_losses        


    def collect_ranking_loss(self, data, pred_pose):
        energy = self.get_energy(data=data, pose_samples=pred_pose, mode='train', extract_pts_feature=False)
        bs = pred_pose.shape[0]
        repeat_num = pred_pose.shape[1]
        repeated_data = {}
        
        """ Repeat input data, [bs, ...] to [bs*repeat_num, ...] """
        for key in ['gt_pose', 'id', 'handle_visibility']:            
            repeat_list = np.ones(len(data[key].shape) + 1, dtype=np.int8).tolist()
            repeat_list[1] = repeat_num
            repeated_data[key] = data[key].unsqueeze(1).repeat(repeat_list).view(bs*repeat_num, -1)
            
        """ Calculate metrics """ 
        res = pred_pose.reshape(bs*repeat_num, -1)
        rot_error, trans_error = get_metrics(
            res,
            repeated_data['gt_pose'],
            class_ids=repeated_data['id'],
            synset_names=self.cfg.synset_names,
            gt_handle_visibility=repeated_data['handle_visibility'],
            pose_mode = self.cfg.pose_mode,
            o2c_pose=self.cfg.o2c_pose,
        )

        metrics = torch.cat(
            (torch.from_numpy(rot_error).to(res.device).unsqueeze(-1), 
             torch.from_numpy(trans_error).to(res.device).unsqueeze(-1)), 
            dim=-1
        )
        metrics = metrics.reshape(bs, repeat_num, -1)    # [bs, repeat_num, pose_dim]
        sorted_energy = sort_results(energy, metrics)
        losses = {'ranking': ranking_loss(sorted_energy)}   
        return losses
          
                   
    def train_energy_func(self, data, pose_samples):
        ''' One step of training '''
        self.net.train()
        self.is_testing = False
        
        data['pts_feat'] = self.net(data, mode='pts_feature')
        self.pts_feature = True
        
        score_losses = self.collect_score_loss(data)
        ranking_losses = self.collect_ranking_loss(data, pose_samples)
        gf_losses = {**score_losses, **ranking_losses}
        
        self.update_network(gf_losses)
        self.record_losses(gf_losses, 'train')
        self.record_lr()
        
        self.ema.update(self.net.parameters())
        if self.cfg.ema_rate > 0 and self.clock.step % 5 == 0:
            ema_losses = self.collect_ema_loss(data)
            self.record_losses(ema_losses, 'train')
        self.pts_feature = False
        return gf_losses
        
        
    def train_score_func(self, data, teacher_model=None):
        """ One step of training """
        self.net.train()
        self.is_testing = False
        
        data['pts_feat'] = self.net(data, mode='pts_feature')
        with torch.no_grad():
            if teacher_model is not None:
                teacher_model.eval()
            pts_feat_teacher = None if teacher_model is None else teacher_model(data, mode='pts_feature')
        self.pts_feature = True
        gf_losses = self.collect_score_loss(data, teacher_model, pts_feat_teacher)
        
        self.update_network(gf_losses)
        self.record_losses(gf_losses, 'train')
        self.record_lr()
        
        self.ema.update(self.net.parameters())
        if self.cfg.ema_rate > 0 and self.clock.step % 5 == 0:
            ema_losses = self.collect_ema_loss(data)
            self.record_losses(ema_losses, 'train')
        self.pts_feature = False
        return gf_losses
    
    
    def train_func(self, data, pose_samples=None, gf_mode='score', teacher_model=None):
        if gf_mode in ['score', 'energy_wo_ranking']:
            losses = self.train_score_func(data, teacher_model)
        elif gf_mode == 'energy':
            losses = self.train_energy_func(data, pose_samples)
        else:
            raise NotImplementedError
        return losses
    
    
    def eval_score_func(self, data, data_mode):
        self.is_testing = True
        self.net.eval()
        self.ema.store(self.net.parameters())
        self.ema.copy_to(self.net.parameters())
        with torch.no_grad():
            data['pts_feat'] = self.net(data, mode='pts_feature')
            self.pts_feature = True
            in_process_sample_list = []
            res_list = []
            sampler_mode_list = self.cfg.sampler_mode
            for sampler in sampler_mode_list:
                in_process_sample, res = self.net(data, mode=f'{sampler}_sample')
                in_process_sample_list.append(in_process_sample)
                res_list.append(res)
            
            metrics = []
            for res_item, sampler_item in zip(res_list, sampler_mode_list):
                metric = self.collect_metric(res_item, data['gt_pose'], data['id'], data['handle_visibility'])
                metrics.append(metric)
                self.record_metrics(metric, sampler_item, data_mode)
            self.visualize_batch(data, res_list, sampler_mode_list, data_mode)
            if self.cfg.save_video:
                save_path = self.model_dir.replace('ckpts', 'inference_results')
                save_path = os.path.join(
                    save_path, 
                    data_mode + '_' + sampler_item + '_' + str(self.cfg.sampling_steps),
                    f'epoch_{str(self.clock.epoch)}'
                )
                print('Saving videos and images...')
                test_time_visulize(save_path, data, res, in_process_sample, self.cfg.pose_mode, self.cfg.o2c_pose)
            self.pts_feature = False
        self.ema.restore(self.net.parameters())

        return metrics, sampler_mode_list
    
    
    def eval_energy_func(self, data, data_mode, pose_samples):
        self.is_testing = True
        self.net.eval()

        with torch.no_grad(): 
            data['pts_feat'] = self.net(data, mode='pts_feature')
            self.pts_feature = True
            
            score_losses = self.collect_score_loss(data)
            ranking_losses = self.collect_ranking_loss(data, pose_samples)
            gf_losses = {**score_losses, **ranking_losses}
            for k, v in ranking_losses.items():
                if not k == 'item':
                    self.writer.add_scalar(f'{data_mode}/loss_{k}', v, self.clock.epoch)
            
        return gf_losses, None


    def eval_func(self, data, data_mode, pose_samples=None, gf_mode='score'):
        if gf_mode in ['score', 'energy_wo_ranking']:
            metrics, sampler_mode_list = self.eval_score_func(data, data_mode)
        elif gf_mode == 'energy':
            losses = self.eval_energy_func(data, data_mode, pose_samples)
        else:
            raise NotImplementedError
        
    
    def test_func(self, data, batch_id):
        self.is_testing = True
        self.net.eval()
        
        with torch.no_grad():
            data['pts_feat'] = self.net(data, mode='pts_feature')
            self.pts_feature = True            
            in_process_sample, res = self.net(data, mode=f'{self.cfg.sampler_mode[0]}_sample')
            sampler_item = self.cfg.sampler_mode[0]
            results = {
                'pred_pose': res,
                'gt_pose': data['gt_pose'],
                'cls_id': data['id'],
                'handle_visibility': data['handle_visibility'],
                # 'path': data['path'],
            }
            metrics = self.collect_metric(res, data['gt_pose'], data['id'], data['handle_visibility'])
            if self.cfg.save_video:
                save_path = self.model_dir.replace('ckpts', 'inference_results')
                save_path = os.path.join(
                    save_path, 
                    self.cfg.test_source + '_' + sampler_item + '_' + str(self.cfg.sampling_steps), 
                    f'eval_num_{str(batch_id)}'
                )
                print('Saving videos and images...')
                test_time_visulize(save_path, data, res, in_process_sample, self.cfg.pose_mode, self.cfg.o2c_pose)
            self.pts_feature = False
            
        return metrics, sampler_item, results


    def pred_func(self, data, repeat_num, save_path='./visualization_results', return_average_res=False, init_x=None, T0=None, return_process=False):

        self.is_testing = True
        self.net.eval()
        
        with torch.no_grad():
            data['pts_feat'] = self.net(data, mode='pts_feature')
            bs = data['pts'].shape[0]
            self.pts_feature = True
            
            ''' Repeat input data, [bs, ...] to [bs*repeat_num, ...] '''
            repeated_data = {}
            for key in data.keys():
                data_shape = [item for item in data[key].shape]
                repeat_list = np.ones(len(data_shape) + 1, dtype=np.int8).tolist()
                repeat_list[1] = repeat_num
                repeated_data[key] = data[key].unsqueeze(1).repeat(repeat_list)
                data_shape[0] = bs*repeat_num
                repeated_data[key] = repeated_data[key].view(data_shape)
            repeated_init_x = None if init_x is None else init_x.unsqueeze(1).repeat(1, repeat_num, 1).view(bs*repeat_num, -1)
            
            ''' Inference '''
            in_process_sample, res = self.net(repeated_data, mode=f'{self.cfg.sampler_mode[0]}_sample', init_x=repeated_init_x, T0=T0)
            pred_pose = res.reshape(bs, repeat_num, -1)
            in_process_sample = in_process_sample.reshape(bs, repeat_num, in_process_sample.shape[1], -1)
            
            ''' Save video '''
            if self.cfg.save_video and not save_path is None:
                print('Saving videos and images...')
                exists_or_mkdir(save_path)
                test_time_visulize(save_path, data, pred_pose[:, 0], in_process_sample[:, 0], self.cfg.pose_mode, self.cfg.o2c_pose)

            self.pts_feature = False
            
            ''' Calculate the average results '''
            if return_average_res:
                rot_matrix = get_rot_matrix(res[:, :-3], self.cfg.pose_mode)
                quat_wxyz = pytorch3d.transforms.matrix_to_quaternion(rot_matrix)
                res_q_wxyz = torch.cat((quat_wxyz, res[:, -3:]), dim=-1)
                pred_pose_q_wxyz = res_q_wxyz.reshape(bs, repeat_num, -1)    # [bs, repeat_num, pose_dim]        
                
                average_pred_pose_q_wxyz = torch.zeros((bs, 7)).to(pred_pose_q_wxyz.device)
                average_pred_pose_q_wxyz[:, :4] = average_quaternion_batch(pred_pose_q_wxyz[:, :, :4])
                average_pred_pose_q_wxyz[:, 4:] = torch.mean(pred_pose_q_wxyz[:, :, 4:], dim=1)
                if return_process:
                    return pred_pose, pred_pose_q_wxyz, average_pred_pose_q_wxyz, in_process_sample
                else:
                    return pred_pose, pred_pose_q_wxyz, average_pred_pose_q_wxyz
            else:
                if return_process:
                    return [pred_pose, in_process_sample]
                else:
                    return pred_pose


    def get_energy(self, data, pose_samples, T=None, mode='test', extract_pts_feature=True):
        if mode == 'train':
            self.is_testing = False
            self.net.train()
        elif mode == 'test':
            self.is_testing = True
            self.net.eval()
        else:
            raise NotImplementedError
        
        """ get pts feature """
        bs = pose_samples.shape[0]
        repeat_num = pose_samples.shape[1]
        if mode == 'train':
            pts_feat = data['pts_feat'] if extract_pts_feature == False else self.net(data, mode='pts_feature')
        elif mode == 'test':
            with torch.no_grad():
                pts_feat = data['pts_feat'] if extract_pts_feature == False else self.net(data, mode='pts_feature')
        self.pts_feature = True
        
        """ repeat pts feature """
        pts_feat_shape = [item for item in pts_feat.shape]
        repeat_list = np.ones(len(pts_feat_shape) + 1, dtype=np.int8).tolist()
        repeat_list[1] = repeat_num
        repeated_pts_feat = pts_feat.unsqueeze(1).repeat(repeat_list).view(bs*repeat_num, -1)
        
        """ get input data of energynet """
        energy_input_data = {
            'pts_feat': repeated_pts_feat,
            'sampled_pose': pose_samples.clone().view(bs*repeat_num, -1).type_as(repeated_pts_feat),
        }
        if T is not None:
            energy_input_data['t'] = torch.ones(bs*repeat_num, 1).type_as(repeated_pts_feat)*T
        else:
            T_min = 1e-5
            T_max = 1e-4
            T_samples = torch.randint(int(T_min*1e5), int(T_max*1e5), (bs, 1)).type_as(repeated_pts_feat) / 1e5
            T_samples = T_samples.repeat([1, repeat_num]).view(bs*repeat_num, 1)
            energy_input_data['t'] = T_samples
        
        pts_center_shape = [item for item in data['pts_center'].shape]
        repeat_list = np.ones(len(pts_center_shape) + 1, dtype=np.int8).tolist()
        repeat_list[1] = repeat_num
        repeated_pts_center = data['pts_center'].unsqueeze(1).repeat(repeat_list)
        repeated_pts_center = repeated_pts_center.view(bs*repeat_num, -1)
        energy_input_data['sampled_pose'][:, -3:] -= repeated_pts_center
        
        if mode == 'train':
            energy = self.net(energy_input_data, mode='energy')
            energy = energy.reshape(bs, repeat_num, -1)
    
        elif mode == 'test':
            with torch.no_grad():
                energy = self.net(energy_input_data, mode='energy')
                energy = energy.reshape(bs, repeat_num, -1)
            
        return energy
    

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), 
                max_norm=self.cfg.grad_clip
            )
        self.optimizer.step()


    def update_learning_rate(self):
        """record and update learning rate"""
        # self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if self.clock.step <= self.cfg.warmup:
            self.optimizer.param_groups[-1]['lr'] = self.base_lr / self.cfg.warmup * self.clock.step
        # elif not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 20.0:
        elif not self.optimizer.param_groups[-1]['lr'] < 1e-4:
            self.scheduler.step()


    def record_losses(self, loss_dict, mode='train'):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        for k, v in losses_values.items():
            self.writer.add_scalar(f'{mode}/{k}', v, self.clock.step)
    
    
    def record_metrics(self, metric, sampler_mode, mode='val'):
        """record metric to tensorboard"""
        rot_error = metric['rot_error']
        trans_error = metric['trans_error']
        
        for k, v in rot_error.items():
            if not k == 'item':
                self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_rot_error', v, self.clock.epoch)
        for k, v in trans_error.items():
            if not k == 'item':
                self.writer.add_scalar(f'{mode}/{sampler_mode}_{k}_trans_error', v, self.clock.epoch)
 

    def record_lr(self):
        self.writer.add_scalar('learing_rate', self.optimizer.param_groups[0]['lr'], self.clock.step)
    
 
    def record_info(self, infos_dict, mode='train', type='scalar'):
        """record loss to tensorboard""" 
        for k, v in infos_dict.items():
            if type == 'scalar':
                self.writer.add_scalar(f'{mode}/{k}', v, self.clock.step)
            elif type == 'scalars':
                self.writer.add_scalars(f'{mode}/{k}', v, self.clock.step)
            elif type == 'image':
                self.writer.add_image(f'{mode}/{k}', v, self.clock.step)
            else:
                raise NotImplementedError
    
    
    def visualize_batch(self, data, res, sampler_mode, mode):
        """write visualization results to tensorboard writer"""
        for res_item, sampler_item in zip(res, sampler_mode):
            pts = torch.cat((data['pts'], data['pts_color']), dim=2)
            if 'color' in data.keys():
                grid_image, _ = create_grid_image(pts, res_item, data['gt_pose'], data['color'], self.cfg.pose_mode, self.cfg.o2c_pose)
            else:
                grid_image, _ = create_grid_image(pts, res_item, data['gt_pose'], None, self.cfg.pose_mode, self.cfg.o2c_pose)
            self.writer.add_image(f'{mode}/vis_{sampler_item}', grid_image, self.clock.epoch)          
    
    
    def collect_metric(self, pred_pose, gt_pose, cat_ids, gt_handle_visibility):
        rot_error, trans_error = get_metrics(
            pred_pose.type_as(gt_pose),
            gt_pose,
            class_ids=cat_ids,
            synset_names=self.cfg.synset_names,
            gt_handle_visibility=gt_handle_visibility,
            pose_mode = self.cfg.pose_mode,
            o2c_pose=self.cfg.o2c_pose,
        )
        rot_error = {
            'mean': np.mean(rot_error),
            'median': np.median(rot_error),
            'item': rot_error,
        }
        trans_error = {
            'mean': np.mean(trans_error),
            'median': np.median(trans_error),
            'item': trans_error,
        }
        error = {'rot_error': rot_error,
                 'trans_error': trans_error}
        return error


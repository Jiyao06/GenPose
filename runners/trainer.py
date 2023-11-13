import sys
import os
import argparse
import pickle
import time
import json
import numpy as np
import torch
import pytorch3d
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ipdb import set_trace
from tqdm import tqdm


# from datasets.datasets_nocs import get_data_loaders_from_cfg, process_batch
from datasets.datasets_genpose import get_data_loaders_from_cfg, process_batch
from networks.posenet_agent import PoseNet 
from configs.config import get_config
from utils.misc import exists_or_mkdir
from utils.genpose_utils import merge_results
from utils.misc import average_quaternion_batch, parallel_setup, parallel_cleanup
from utils.metrics import get_metrics, get_rot_matrix
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image

   
def prediction(cfg, dataloader, agent):
    calc_confidence = False
    calc_energy = True
    if len(cfg.sampler_mode) != 1:
        raise NotImplementedError
    
    results = {}
    for index, test_batch in enumerate(tqdm(dataloader)):
        # inference a small batch samples
        if index > cfg.max_eval_num:
            break
        
        batch_sample = process_batch(
            batch_sample = test_batch, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
        )
        gt_rot_matrix = get_rot_matrix(batch_sample['gt_pose'][:, :-3], cfg.pose_mode)
        gt_quat_wxyz = pytorch3d.transforms.matrix_to_quaternion(gt_rot_matrix)
                
        pred_pose, average_pred_pose, choosed_pred_pose, energy = agent.pred_func(
            data=batch_sample, 
            repeat_num=cfg.repeat_num, 
            calc_confidence=calc_confidence,
            calc_energy=calc_energy) # [bs, repeat_num, 7]
        
        result = {
            'pred_pose': pred_pose,
            'average_pred_pose': average_pred_pose,
            'choosed_pred_pose': choosed_pred_pose,
            'gt_pose': torch.cat((gt_quat_wxyz, batch_sample['gt_pose'][:, -3:]), dim=-1),
            'pts': batch_sample['pts'],
            # 'log_likelihoods': log_likelihoods, # [bs, repeat_num]
        }
        
        rot_error, trans_error = get_metrics(
            result['average_pred_pose'],
            result['gt_pose'],
            class_ids=batch_sample['id'],
            synset_names=cfg.synset_names,
            gt_handle_visibility=batch_sample['handle_visibility'],
            pose_mode = 'quat_wxyz',
            o2c_pose=cfg.o2c_pose,
        )
        result['metrics'] = torch.cat((torch.from_numpy(rot_error).reshape(-1, 1), torch.from_numpy(trans_error).reshape(-1, 1)), dim=-1)
        print('mean error: ', torch.mean(result['metrics'], dim=0))
        print('median error: ', torch.median(result['metrics'], dim=0).values)

        rot_error, trans_error = get_metrics(
            result['choosed_pred_pose'],
            result['gt_pose'],
            class_ids=batch_sample['id'],
            synset_names=cfg.synset_names,
            gt_handle_visibility=batch_sample['handle_visibility'],
            pose_mode = 'quat_wxyz',
            o2c_pose=cfg.o2c_pose,
        )
        result['metrics'] = torch.cat((torch.from_numpy(rot_error).reshape(-1, 1), torch.from_numpy(trans_error).reshape(-1, 1)), dim=-1)
        print('mean error: ', torch.mean(result['metrics'], dim=0))
        print('median error: ', torch.median(result['metrics'], dim=0).values)        
        # indices = torch.arange(0, result['pred_pose'].shape[0]).view(1, -1).to(result['pred_pose'].device)
        # max_likelihood_index = torch.argmax(result['log_likelihoods'], dim=-1).view(1, -1)
        # rot_error, trans_error = get_metrics(
        #     result['pred_pose'][torch.cat((indices, max_likelihood_index), dim=0).cpu().numpy().tolist()],
        #     result['gt_pose'],
        #     class_ids=batch_sample['id'],
        #     synset_names=cfg.synset_names,
        #     gt_handle_visibility=batch_sample['handle_visibility'],
        #     pose_mode = 'quat_wxyz',
        #     o2c_pose=cfg.o2c_pose,
        # )
        # result['metrics'] = torch.cat((torch.from_numpy(rot_error).reshape(-1, 1), torch.from_numpy(trans_error).reshape(-1, 1)), dim=-1)
        # print('mean error: ', torch.mean(result['metrics'], dim=0))
        # print('median error: ', torch.median(result['metrics'], dim=0).values)         
        # set_trace()
        
        if index == 0:
            results = result
        else:
            for key in results.keys():
                if not results[key] == None:
                    results[key] = torch.cat((results[key], result[key]), dim=0)
    
    ''' results visualization '''
    for i in range(results['pred_pose'].shape[0]):
        gt_rot = results['gt_pose'][i][:-3].unsqueeze(0)
        pred_rot = results['pred_pose'][i][:, :-3]
        choosed_pred_rot = results['choosed_pred_pose'][i][:-3].unsqueeze(0)
        average_pred_rot = results['average_pred_pose'][i][:-3].unsqueeze(0)
        if calc_confidence == True:
            # confidence = torch.tanh(results['log_likelihoods'][i])
            index = torch.argmax(results['log_likelihoods'][i])
            max_likelihood_pred_pose = pred_rot[index].unsqueeze(0)
            # pred_pose = torch.cat((average_pred_pose, max_likelihood_pred_pose), dim=0)
        ''' ToDo: render pointcloud '''
        grid_iamge, _ = create_grid_image(
            results['pts'][i].unsqueeze(0), 
            results['average_pred_pose'][i].unsqueeze(0), 
            results['gt_pose'][i].unsqueeze(0), 
            None, 
            pose_mode='quat_wxyz', 
            inverse_pose=cfg.o2c_pose,
        )
        ''' so3 distribution visualization '''
        visualize_so3(
            save_path='./so3_distribution.png', 
            pred_rotations=get_rot_matrix(pred_rot).cpu().numpy(),
            pred_rotation=get_rot_matrix(average_pred_rot).cpu().numpy(),
            gt_rotation=get_rot_matrix(gt_rot).cpu().numpy(),
            image=grid_iamge,
            # probabilities=confidence
            )
        set_trace()
        grid_iamge, _ = create_grid_image(
            results['pts'][i].unsqueeze(0), 
            results['choosed_pred_pose'][i].unsqueeze(0), 
            results['gt_pose'][i].unsqueeze(0), 
            None, 
            pose_mode='quat_wxyz', 
            inverse_pose=cfg.o2c_pose,
        )
        visualize_so3(
            save_path='./so3_distribution.png', 
            pred_rotations=get_rot_matrix(pred_rot).cpu().numpy(),
            pred_rotation=get_rot_matrix(choosed_pred_rot).cpu().numpy(),
            gt_rotation=get_rot_matrix(gt_rot).cpu().numpy(),
            image=grid_iamge,
            # probabilities=confidence
            )
        set_trace()
    
    return results
    

def inference(cfg, dataloader, agent):
    if len(cfg.sampler_mode) != 1:
        raise NotImplementedError
    
    repeat_num = cfg.repeat_num
    metrics = {}
    for i in range(repeat_num):
        epoch_rot_error = np.array([])
        epoch_trans_error = np.array([])
        epoch_results = {}
        pbar = tqdm(dataloader)
        pbar.set_description(f'NUM[{i+1}/{repeat_num}]')
        for index, test_batch in enumerate(pbar):
            # inference a small batch samples
            if index > cfg.max_eval_num:
                break
            
            batch_sample = process_batch(
                batch_sample = test_batch, 
                device=cfg.device, 
                pose_mode=cfg.pose_mode,
            )
            
            batch_metrics, sampler_mode, batch_results = agent.test_func(batch_sample, index)
            epoch_rot_error = np.concatenate([epoch_rot_error, batch_metrics['rot_error']['item']])
            epoch_trans_error = np.concatenate([epoch_trans_error, batch_metrics['trans_error']['item']])
            epoch_results = merge_results(epoch_results, batch_results)
            pbar.set_postfix({
                'MEAN_ROT_ERROR: ': batch_metrics['rot_error']['item'].mean(),
                'MEAN_TRANS_ERROR: ': batch_metrics['trans_error']['item'].mean()
            })
            
        pbar.set_postfix({
            'MEAN_ROT_ERROR: ': epoch_rot_error.mean(),
            'MEAN_TRANS_ERROR: ': epoch_trans_error.mean()
        })
        print("MEAM ROTATION ERROR: ", epoch_rot_error.mean())
        print("MEAN TRANSLATION ERROR: ", epoch_trans_error.mean())
        print("MEDIAN ROTATION ERROR: ", np.median(epoch_rot_error))
        print("MEDIAN TRANSLATION ERROR: ", np.median(epoch_trans_error))
        
        error = np.concatenate([epoch_rot_error[..., np.newaxis], epoch_trans_error[..., np.newaxis]], axis=1)
        metrics[f'index_{i}'] = error.tolist()
        
        if i == 0:
            results = epoch_results
            results['pred_pose'] = results['pred_pose'].unsqueeze(1)
        else:
            results['pred_pose'] = torch.cat([results['pred_pose'], epoch_results['pred_pose'].unsqueeze(1)], dim=1)
    
    ''' Save metrics and results '''
    save_path = agent.model_dir.replace('ckpts', 'inference_results')
    save_path = os.path.join(
        save_path,
        cfg.test_source + '_' + sampler_mode + '_' + str(cfg.sampling_steps)
    )
    exists_or_mkdir(save_path)
    metrics_save_path = os.path.join(save_path, 'metrics.json')
    
    with open(metrics_save_path, 'w') as f:
        f.write(json.dumps(metrics, indent=1))
    f.close()        
    
    ''' Save results '''
    results_save_path = os.path.join(save_path, 'results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(results, f)
    f.close()       
    return results_save_path


def evaluation(cfg):
    results_path = cfg.results_path
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print("Merging results...")
    results['average_pred_pose'] = torch.zeros_like(results['gt_pose'])
    results['average_pred_pose'][:, :4] = average_quaternion_batch(results['pred_pose'][:, :, :4])
    results['average_pred_pose'][:, 4:] = torch.mean(results['pred_pose'][:, :, 4:], dim=1)
    
    print("Caculating metrics...")
    rot_error, trans_error = get_metrics(
        results['average_pred_pose'],
        results['gt_pose'],
        class_ids=results['cls_id'],
        synset_names=cfg.synset_names,
        gt_handle_visibility=results['handle_visibility'],
        pose_mode=cfg.pose_mode,
        o2c_pose=cfg.o2c_pose,
    )
    
    error = np.concatenate([rot_error[..., np.newaxis], trans_error.T[..., np.newaxis]], axis=1)
    results_save_path = results_path.replace('results.pkl', 'average_results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(error, f)
    
    print("Mean Rotation Error: ", np.mean(rot_error))
    print("Median Rotation Error: ", np.median(rot_error))
    print("Mean Translation Error: ", np.mean(trans_error))
    print("Median Translation Error: ", np.median(trans_error))


def train_score(cfg, train_loader, val_loader, score_agent, teacher_model=None):
    """ Train score network or energe network without ranking
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        score_agent (torch.nn.Module): score network or energy network without ranking
    Returns:
    """
    
    for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
        ''' train '''
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        pbar = tqdm(train_loader)
        for i, batch_sample in enumerate(pbar):
            
            ''' warm up'''
            if score_agent.clock.step < cfg.warmup:
                score_agent.update_learning_rate()
                
            ''' load data '''
            batch_sample = process_batch(
                batch_sample = batch_sample, 
                device=cfg.device, 
                pose_mode=cfg.pose_mode, 
                PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
            )
            
            ''' train score or energe without feedback'''
            losses = score_agent.train_func(data=batch_sample, gf_mode='score', teacher_model=teacher_model)
            
            pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
            score_agent.clock.tick()
        
        ''' updata learning rate and clock '''
        # if epoch >= 50 and epoch % 50 == 0:
        score_agent.update_learning_rate()
        score_agent.clock.tock()

        ''' start eval '''
        if score_agent.clock.epoch % cfg.eval_freq == 0:   
            data_loaders = [train_loader, val_loader]    
            data_modes = ['train', 'val']   
            for i in range(len(data_modes)):
                test_batch = next(iter(data_loaders[i]))
                data_mode = data_modes[i]
                test_batch = process_batch(
                    batch_sample=test_batch,
                    device=cfg.device,
                    pose_mode=cfg.pose_mode,
                    mini_batch_size=cfg.mini_bs,
                )
                score_agent.eval_func(test_batch, data_mode)
                
            ''' save (ema) model '''
            score_agent.save_ckpt()


def train_energy(cfg, train_loader, val_loader, energy_agent, score_agent=None, ranking=False, distillation=False):
    """ Train score network or energe network without ranking
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        energy_agent (torch.nn.Module): energy network with ranking
        score_agent (torch.nn.Module): score network
        ranking (bool): train energy network with ranking or not
    Returns:
    """
    if ranking is False:
        teacher_model = None if not distillation else score_agent.net
        train_score(cfg, train_loader, val_loader, energy_agent, teacher_model)
    else:
        for epoch in range(energy_agent.clock.epoch, cfg.n_epochs):
            torch.cuda.empty_cache()
            pbar = tqdm(train_loader)
            for i, batch_sample in enumerate(pbar):
                
                ''' warm up '''
                if energy_agent.clock.step < cfg.warmup:
                    energy_agent.update_learning_rate()
                    
                ''' get data '''
                batch_sample = process_batch(
                    batch_sample = batch_sample, 
                    device=cfg.device, 
                    pose_mode=cfg.pose_mode, 
                    PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
                )
                
                ''' get pose samples from pretrained score network '''
                pred_pose = score_agent.pred_func(data=batch_sample, repeat_num=5, save_path=None)
                
                ''' train energy '''
                losses = energy_agent.train_func(data=batch_sample, pose_samples=pred_pose, gf_mode='energy')
                pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
                
                energy_agent.clock.tick()
            energy_agent.update_learning_rate()
            energy_agent.clock.tock()

            ''' start eval '''
            if energy_agent.clock.epoch % cfg.eval_freq == 0:   
                data_loaders = [train_loader, val_loader]    
                data_modes = ['train', 'val']   
                for i in range(len(data_modes)):
                    test_batch = next(iter(data_loaders[i]))
                    data_mode = data_modes[i]
                    test_batch = process_batch(
                        batch_sample=test_batch,
                        device=cfg.device,
                        pose_mode=cfg.pose_mode,
                        mini_batch_size=cfg.mini_bs,
                    )
                    
                    ''' get pose samples from pretrained score network '''
                    pred_pose = score_agent.pred_func(data=test_batch, repeat_num=5, save_path=None)
                    energy_agent.eval_func(test_batch, data_mode, None, 'score')
                    energy_agent.eval_func(test_batch, data_mode, pred_pose, 'energy')
                
                ''' save (ema) model '''
                energy_agent.save_ckpt()


def main():
    # load config
    cfg = get_config()
    if len(cfg.results_path):
        print("Start evaluate ...")
        evaluation(cfg)
        print("Evaluate finished!")
        exit()
    
    ''' Init data loader '''
    if not (cfg.eval or cfg.pred):
        data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['train', 'val', 'test'])
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
        print('train_set: ', len(train_loader))
        print('val_set: ', len(val_loader))
        print('test_set: ', len(test_loader))
    else:
        data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['test'])
        test_loader = data_loaders['test_loader']   
        print('test_set: ', len(test_loader))
  
    
    ''' Init trianing agent and load checkpoints'''
    if cfg.agent_type == 'score':
        cfg.posenet_mode = 'score'
        score_agent = PoseNet(cfg)
        tr_agent = score_agent
        
    elif cfg.agent_type == 'energy':
        cfg.posenet_mode = 'energy'
        energy_agent = PoseNet(cfg)
        if cfg.pretrained_score_model_path is not None:
            energy_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
            energy_agent.net.pose_score_net.output_zero_initial()
        if cfg.distillation is True:
            cfg.posenet_mode = 'score'
            score_agent = PoseNet(cfg)
            score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
            cfg.posenet_mode = 'energy'
        tr_agent = energy_agent
        
    elif cfg.agent_type == 'energy_with_ranking':
        cfg.posenet_mode = 'score'
        score_agent = PoseNet(cfg)    
        cfg.posenet_mode = 'energy'
        energy_agent = PoseNet(cfg)
        score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        if cfg.pretrained_energy_model_path:
            energy_agent.load_ckpt(model_dir=cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
        tr_agent = energy_agent
    
    else:
        raise NotImplementedError
    
    ''' Load checkpoints '''
    load_model_only = False if cfg.use_pretrain else True
    if cfg.use_pretrain or cfg.eval or cfg.pred:
        tr_agent.load_ckpt(model_dir=cfg.pretrained_model_path, model_path=True, load_model_only=load_model_only)        
                
    
    ''' Start testing loop'''
    if cfg.eval:
        print("Start inference ...")
        inference(cfg, test_loader, tr_agent)
        print("Inference finished")
        exit()
    if cfg.pred:
        print("Start prediction ...")
        prediction(cfg, test_loader, tr_agent)
        print("Prediction finished")
        exit()
        
        
    ''' Start training loop '''
    if cfg.agent_type == 'score':
        train_score(cfg, train_loader, val_loader, tr_agent)
    elif cfg.agent_type == 'energy':
        if cfg.distillation:
            train_energy(cfg, train_loader, val_loader, tr_agent, score_agent, False, True)
        else:
            train_energy(cfg, train_loader, val_loader, tr_agent)
    elif cfg.agent_type == 'energy_with_ranking':
        train_energy(cfg, train_loader, val_loader, tr_agent, score_agent, True)
    
if __name__ == '__main__':
    main()



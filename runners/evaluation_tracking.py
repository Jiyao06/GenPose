import os
import sys
import time
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch3d
import shutil
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ipdb import set_trace
from utils.misc import exists_or_mkdir, get_rot_matrix, average_quaternion_batch
from utils.datasets_utils import crop_resize_by_warp_affine, get_2d_coord_np
from utils.sgpa_utils import load_depth, get_bbox, compute_RT_errors
from utils.tracking_utils import add_noise_to_RT
from networks.posenet_agent import PoseNet
from networks.reward import sort_poses_by_energy
from configs.config import get_config


''' load config '''
cfg = get_config()


''' create checkpoint list '''
scorenet_ckpt_path = f'./results/ckpts/{cfg.score_model_dir}'
energynet_ckpt_path = f'./results/ckpts/{cfg.energy_model_dir}'


''' create result dir '''
result_dir = cfg.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
inference_res_root_dir = os.path.join(result_dir, 'evaluation_results')
inference_res_dir = os.path.join(inference_res_root_dir, f'{cfg.test_source}_repeat_{cfg.eval_repeat_num}')
segmentation_results_path = os.path.join(inference_res_root_dir, 'segmentation_results_{}.pkl'.format(cfg.test_source))
exists_or_mkdir(inference_res_dir)

file_path = 'Real/test_list.txt'
cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
camera_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)


if not os.path.exists(result_dir):
    os.makedirs(result_dir)


xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0


def cal_average_sRT(sRT, selected_num):
    bs = sRT.shape[0]
    selected_sRT = sRT[:, :selected_num, :, :]
    
    ''' calculate averaged sRT of selected sRT '''
    reshaped_selected_sRT = selected_sRT.reshape(bs*selected_num, 4, 4)
    reshaped_selected_sRT = torch.from_numpy(reshaped_selected_sRT).cuda()
    quat_wxyz = pytorch3d.transforms.matrix_to_quaternion(reshaped_selected_sRT[:, :3, :3])
    quat_wxyz = torch.cat((quat_wxyz, reshaped_selected_sRT[:, :3, 3]), dim=-1)
    quat_wxyz = quat_wxyz.reshape(bs, selected_num, -1)
    
    average_pred_pose = torch.zeros((quat_wxyz.shape[0], quat_wxyz.shape[-1])).to(quat_wxyz.device)
    average_pred_pose[:, :4] = average_quaternion_batch(quat_wxyz[:, :, :4])
    average_pred_pose[:, 4:] = torch.mean(quat_wxyz[:, :, 4:], dim=1)
    average_sRT = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1)
    average_sRT[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(average_pred_pose[:, :4])
    average_sRT[:, :3, 3] = average_pred_pose[:, 4:]
    return average_sRT
    
    
def depth_to_pcl(depth, K, xymap, mask):
    K = K.reshape(-1)
    cx, cy, fx, fy = K[2], K[5], K[0], K[4]
    depth = depth.reshape(-1).astype(np.float32)
    valid = ((depth > 0) * mask.reshape(-1)) > 0
    depth = depth[valid]
    x_map = xymap[0].reshape(-1)[valid]
    y_map = xymap[1].reshape(-1)[valid]
    real_x = (x_map - cx) * depth / fx
    real_y = (y_map - cy) * depth / fy
    pcl = np.stack((real_x, real_y, depth), axis=-1)
    return pcl.astype(np.float32)


def sample_points(pcl, n_pts):
    """ Down sample the point cloud using farthest point sampling.

    Args:
        pcl (torch tensor or numpy array):  NumPoints x 3
        num (int): target point number
    """
    total_pts_num = pcl.shape[0]
    if total_pts_num < n_pts:
        pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
    elif total_pts_num > n_pts:
        ids = np.random.permutation(total_pts_num)[:n_pts]
        pcl = pcl[ids]
    return pcl
    
    
def extract_single_frame_data(path):
    img_path = os.path.join(cfg.data_path, path)
    raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
    raw_rgb = raw_rgb[:, :, ::-1]
    raw_depth = load_depth(img_path)
    im_H, im_W = raw_rgb.shape[0], raw_rgb.shape[1]

    # load mask-rcnn detection results
    img_path_parsing = img_path.split('/')
    mrcnn_path = os.path.join(result_dir, 'mrcnn_results', cfg.test_source, 'results_{}_{}_{}.pkl'.format(
        cfg.test_source.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
    with open(mrcnn_path, 'rb') as f:
        mrcnn_result = cPickle.load(f)
    num_insts = len(mrcnn_result['pred']['class_ids'])
    f_sRT = np.identity(4, dtype=float)[np.newaxis, ...].repeat(num_insts, 0)
    f_size = np.ones((num_insts, 3), dtype=float)

    # prepare frame data
    f_points, f_catId = [], []
    for i in range(num_insts):
        cat_id = mrcnn_result['pred']['class_ids'][i] - 1
        # rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
        rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['pred']['rois'][i])
        mask = np.logical_and(mrcnn_result['pred']['masks'][:, :, i], raw_depth > 0)
        
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        x1, y1, x2, y2 = bbox_xyxy
        # here resize and crop to a fixed size 256 x 256
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
        scale = min(scale, max(im_H, im_W)) * 1.0

        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, cfg.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        mask_target = mask.copy().astype(np.float32)
        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, cfg.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)

        roi_depth = crop_resize_by_warp_affine(
            raw_depth, bbox_center, scale, cfg.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_depth = np.expand_dims(roi_depth, axis=0)

        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            f_sRT[i] = np.identity(4, dtype=float)
            continue
            
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            f_sRT[i] = np.identity(4, dtype=float)
            continue

        pcl_in = depth_to_pcl(roi_depth, camera_intrinsics, roi_coord_2d, roi_mask) / 1000.0
        points = sample_points(pcl_in, cfg.num_points)
        
        mrcnn_result['gt']['poses'][i, :3, :3] = mrcnn_result['gt']['poses'][i, :3, :3] / np.linalg.norm(mrcnn_result['gt']['poses'][i][:3, 0])
        # concatenate instances
        f_points.append(points)
        f_catId.append(cat_id)

    
    return {'gt_handle_visibility': mrcnn_result['gt']['handle_visibility'],
            'gt_pose': mrcnn_result['gt']['poses'],
            'tracked': mrcnn_result['pred']['seen'],
            'model_name': mrcnn_result['gt']['model_list'].tolist(),
            'pts': f_points,
            'cat_id': f_catId}


def pred_pose_batch(score_agent, batch_sample, visualization_save_path, init_x=None, T0=None):
    ''' Predict poses '''
    pred_pose = score_agent.pred_func(
        data=batch_sample, 
        repeat_num=cfg.eval_repeat_num, 
        save_path=visualization_save_path,
        return_average_res=False, 
        init_x=init_x,
        T0=T0
    )        
    return pred_pose


def pred_energy_batch(energy_agent, batch_sample, pred_pose, pose_mode):  
    ''' Predict energy '''
    energy = energy_agent.get_energy(
        data=batch_sample, 
        pose_samples=pred_pose, 
        T=1e-5
    )
    sorted_pred_pose, sorted_energy = sort_poses_by_energy(pred_pose, energy)    
    RTs_all = np.ones((sorted_pred_pose.shape[0], sorted_pred_pose.shape[1], 4, 4))    #[bs, repeat_num, 4, 4]
    for i in range(sorted_pred_pose.shape[1]):
        R = get_rot_matrix(sorted_pred_pose[:, i, :-3], pose_mode)
        T = sorted_pred_pose[:, i, -3:]
        RTs = np.identity(4, dtype=float)[np.newaxis, ...].repeat(R.shape[0], 0)
        RTs[:, :3, :3] = R.cpu().numpy()
        RTs[:, :3, 3] = T.cpu().numpy()
        RTs_all[:, i, :, :] = RTs
    return RTs_all #, sorted_energy.cpu().numpy()


def get_metrics(errors: dict):
    ''' get metrics '''
    cls = {}
    for key in errors.keys():
        cls_name = key.split('_')[0]
        cls[cls_name] = np.array(errors[key]) if cls_name not in cls else np.concatenate([cls[cls_name], np.array(errors[key])], axis=0)
    metrics = {}
    for key in cls.keys():
        cls_errors = np.array(cls[key])
        metrics[key] = {}
        metrics[key]['mean_error'] = np.mean(cls_errors, axis=0)
        metrics[key]['5d5cm_acc'] = np.sum(np.logical_and(cls_errors[:, 0] <= 5, cls_errors[:, 1] <= 5)) / cls_errors.shape[0]
        
    cls_num = len(metrics.keys())
    metrics['all'] = {
        'mean_error': 0,
        '5d5cm_acc': 0
    }
    for key in metrics.keys():
        if key == 'all':
            continue
        metrics['all']['mean_error'] += metrics[key]['mean_error'] / cls_num
        metrics['all']['5d5cm_acc'] += metrics[key]['5d5cm_acc'] / cls_num
        
    return metrics    


def record_message(metrics: dict, write_path: str):
    ''' record message '''
    fw = open(write_path, 'w')
    for key in metrics.keys():
        message = key
        fw.write(message + '\n')
        print(message)
        for sub_key in metrics[key].keys():
            message = '{}: {}'.format(sub_key, metrics[key][sub_key])
            fw.write(message + '\n')
            print(message)
    fw.close()
    
    
def main_tracking(tracking=False, T0=1.0):
    ''' Create evaluation agent '''
    cfg.posenet_mode = 'score'
    score_agent = PoseNet(cfg)  
    cfg.posenet_mode = 'energy'
    energy_agent = PoseNet(cfg)
    
    ''' Load model '''
    score_agent.load_ckpt(model_dir=scorenet_ckpt_path, model_path=True, load_model_only=True)  
    energy_agent.load_ckpt(model_dir=energynet_ckpt_path, model_path=True, load_model_only=True)  
    
    ''' Load data '''
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n')) for line in open(os.path.join(cfg.data_path, file_path))]
    img_list = sorted(img_list)

    ''' create video dir '''
    video_path = {}
    for cls_name in cfg.synset_names:
        video_path[cls_name] = f'{inference_res_dir}/videos/{cfg.test_source}/{cls_name}'
        try:
            shutil.rmtree(video_path[cls_name])
        except Exception as e:
            pass
        if cfg.save_video:
            exists_or_mkdir(video_path[cls_name])
            
    ''' Inference '''
    errors = {}
    results_buffer = {'model_name': [], 'pred_sRT': []}
    
    for path in tqdm(img_list):
        detect_result = extract_single_frame_data(path)
        pts = torch.cuda.FloatTensor(np.array(detect_result['pts']))
        batch_sample = {'pts': pts}
        num_pts = pts.shape[1]
        zero_mean = torch.mean(pts[:, :, :3], dim=1)
        batch_sample['zero_mean_pts'] = copy.deepcopy(pts)
        batch_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
        batch_sample['pts_center'] = zero_mean
        
        if tracking:
            previous_frame_results = results_buffer
            initial_sRT = add_noise_to_RT(torch.tensor(detect_result['gt_pose']).cuda())
            for i, item in enumerate(detect_result['model_name']):
                if item in previous_frame_results['model_name']:
                    index = previous_frame_results['model_name'].index(item)
                    initial_sRT[i] = previous_frame_results['pred_sRT'][index]
            initial_pose = initial_sRT[:, :3, [0, 1, 3]].permute(0, 2, 1).reshape(initial_sRT.shape[0], -1)
            initial_pose[:, -3:] -= batch_sample['pts_center']
        else:
            initial_pose = None
        
        ''' Predict pose and energy'''
        pred_pose = pred_pose_batch(score_agent, batch_sample, video_path, init_x=initial_pose, T0=T0)
        sorted_sRT_all = pred_energy_batch(energy_agent, batch_sample, pred_pose, cfg.pose_mode)
        average_sRT = cal_average_sRT(sorted_sRT_all, max(1, int(0.6 * cfg.eval_repeat_num)))
        
        results_buffer['model_name'] = detect_result['model_name']
        results_buffer['pred_sRT'] = average_sRT
        
        average_sRT = average_sRT.cpu().numpy()
        for i in range(average_sRT.shape[0]):
            RT_errors = compute_RT_errors(
                average_sRT[i], detect_result['gt_pose'][i], detect_result['cat_id'][i], 
                detect_result['gt_handle_visibility'][i], cfg.synset_names)
            if detect_result['model_name'][i] not in errors:
                errors[detect_result['model_name'][i]] = [RT_errors.tolist()]
            else:
                errors[detect_result['model_name'][i]] = errors[detect_result['model_name'][i]] + [RT_errors.tolist()]
            if detect_result['model_name'][i] == 'mug_brown_starbucks_norm' and RT_errors[0] > 50:
                set_trace()

    metrics = get_metrics(errors)
    save_path = f'{inference_res_dir}/results/{cfg.pooling_mode}/{cfg.ranker}'
    exists_or_mkdir(save_path)
    record_message(metrics, f'{save_path}/eval_logs.txt')
     

if __name__ == '__main__':
    tracking = True
    main_tracking(tracking, cfg.T0)


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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ipdb import set_trace
from utils.misc import exists_or_mkdir, get_rot_matrix, average_quaternion_batch, average_quaternion_torch, average_quaternion_numpy
from utils.datasets_utils import crop_resize_by_warp_affine, get_2d_coord_np
from utils.sgpa_utils import load_depth, get_bbox, compute_mAP, plot_mAP, draw_detections

from networks.posenet_agent import PoseNet
from networks.reward import sort_poses_by_energy, ranking_loss
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
    

assert cfg.test_source in ['val', 'real_test']
if cfg.test_source == 'val':
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]], dtype=np.float32)  # [fx, fy, cx, cy]
else:
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
    camera_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)


xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0


def record_results_and_draw_curves(save_path, iou_aps, pose_aps, iou_acc, pose_acc, iou_thres_list, degree_thres_list, shift_thres_list):
    # draw curves
    plot_mAP(iou_aps, pose_aps, save_path['inference_res_dir'], iou_thres_list, degree_thres_list, shift_thres_list, f"{save_path['pooling_mode']}_ratio_{save_path['ratio']}_mAP.png")
    # record results
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    cls_names = cfg.synset_names
    for i in range(1, 8, 1):
        cls_name = 'mean' if i==7 else cls_names[i-1]
        messages = []
        messages.append(f"cls_name: {cls_name}")
        messages.append(f"{save_path['pooling_mode']}_ratio_{save_path['ratio']}")
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[i, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[i, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[i, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[i, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[i, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[i, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[i, degree_10_idx, shift_05_idx] * 100))
        messages.append('Acc:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[i, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[i, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[i, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[i, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[i, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[i, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[i, degree_10_idx, shift_05_idx] * 100))
        if i == 7:
            fw = open(f"{save_path['inference_res_dir']}/eval_logs.txt", 'a')
        else:
            fw = open(f"{save_path['inference_res_dir']}/eval_logs_single_cls.txt", 'a')
        for msg in messages:
            print(msg)
            fw.write(msg + '\n')
        fw.close()
        

def detect_mrcnn_genpose(save_path):
    
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
    
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(cfg.data_path, file_path))]
    # frame by frame test
    t_inference = 0.0
    t_umeyama = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()
    
    test_data = {}
        
    for path in tqdm(img_list):
        img_path = os.path.join(cfg.data_path, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_depth(img_path)
        im_H, im_W = raw_rgb.shape[0], raw_rgb.shape[1]

        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join(result_dir, 'mrcnn_results', cfg.test_source, 'results_{}_{}_{}.pkl'.format(
            cfg.test_source.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        # mrcnn_path = os.path.join(result_dir, 'mrcnn_results', cfg.test_source, 'results_{}_{}_{}.pkl'.format(
        #     cfg.test_source.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_sRT = np.identity(4, dtype=float)[np.newaxis, ...].repeat(num_insts, 0)
        f_size = np.ones((num_insts, 3), dtype=float)

        # prepare frame data
        f_points, f_catId = [], []
        valid_inst = []
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            
            # roi_coord_2d ----------------------------------------------------
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

            valid_inst.append(i)
            pcl_in = depth_to_pcl(roi_depth, camera_intrinsics, roi_coord_2d, roi_mask) / 1000.0
            points = sample_points(pcl_in, cfg.num_points)

            # concatenate instances
            f_points.append(points)
            f_catId.append(cat_id)

        
        
        if len(valid_inst):
            img_count += 1
            inst_count += len(valid_inst)

        # save results
        result = {}
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        result['gt_class_ids'] = gts['class_ids']
        result['gt_bboxes'] = gts['bboxes']
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_handle_visibility'] = gts['handle_visibility']

        result['pred_class_ids'] = mrcnn_result['class_ids']
        # result['pred_bboxes'] = mrcnn_result['rois']
        result['pred_bboxes'] = mrcnn_result['rois']
        result['pred_scores'] = mrcnn_result['scores']
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size
        
        test_data[img_path] = {'result': result,
                               'valid_pts': f_points,
                               'valid_rgb': None,
                               'cat_id': f_catId,
                               'valid_inst': valid_inst}
        
        # image_short_path = '_'.join(img_path_parsing[-3:])
        # save_path = os.path.join(result_dir, 'results_{}.pkl'.format(image_short_path))
        # with open(save_path, 'wb') as f:
        #     cPickle.dump(result, f)
        
    with open(save_path, 'wb') as f:
        cPickle.dump(test_data, f)
    # write statistics
    fw = open(save_path.replace('_results_', '_logs_').replace('.pkl', '.txt'), 'w')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    fw.writelines([msg + '\n' for msg in messages])


def unpack_data(path):
    detect_result_path = path
    with open(detect_result_path, 'rb') as f:
        detect_result = cPickle.load(f)       

    categorized_test_data = {}
    for cat_name in cfg.synset_names:
        categorized_test_data[cat_name] = {'img_path': [],
                                           'pts': [],
                                           'rgb': [],
                                           'cat_id': [],
                                           'inst': [],}
    
    print('Extracting data...')
    for key in tqdm(detect_result.keys()):
        instance_num = detect_result[key]['result']['pred_RTs'].shape[0]
        # detect_result[key]['result']['choosed_pred_RTs'] = copy.deepcopy(detect_result[key]['result']['pred_RTs'])
        detect_result[key]['result']['multi_hypothesis_pred_RTs'] = \
            np.identity(4, dtype=float)[np.newaxis, np.newaxis, ...].repeat(instance_num, axis=0).repeat(cfg.eval_repeat_num, axis=1)
        detect_result[key]['result']['energy'] = \
            np.zeros(2, dtype=float)[np.newaxis, np.newaxis, ...].repeat(instance_num, axis=0).repeat(cfg.eval_repeat_num, axis=1)
        
        # result = detect_result[key]['result']
        valid_pts = detect_result[key]['valid_pts']
        valid_rgb = detect_result[key]['valid_rgb']
        cat_id = detect_result[key]['cat_id']
        valid_inst = detect_result[key]['valid_inst']
        # if len(valid_inst):
        #     f_catId = torch.cuda.LongTensor(f_catId)
        #     f_points = torch.cuda.FloatTensor(f_points)
        #     f_rgb = torch.cuda.FloatTensor(f_rgb)
            
        if len(valid_inst):
            for i in range(len(valid_inst)):
                cat_name = cfg.synset_names[cat_id[i]]
                categorized_test_data[cat_name]['img_path'].append(key)
                categorized_test_data[cat_name]['pts'].append(valid_pts[i])
                categorized_test_data[cat_name]['cat_id'].append(cat_id[i])
                categorized_test_data[cat_name]['inst'].append(valid_inst[i])
                if not valid_rgb is None:
                    categorized_test_data[cat_name]['rgb'].append(valid_rgb[i])
                else:
                    categorized_test_data[cat_name]['rgb'] = None
    return detect_result, categorized_test_data


def pred_pose_batch(score_agent:PoseNet, batch_sample, visualization_save_path, pose_mode, return_process=False):
    ''' Predict poses '''
    pred_results = score_agent.pred_func(
        data=batch_sample, 
        repeat_num=cfg.eval_repeat_num, 
        save_path=visualization_save_path,
        T0=cfg.T0,
        return_average_res=False,
        return_process=return_process
    )
    if return_process:
        pred_pose = pred_results[0]
    else:
        pred_pose = pred_results
    
    ''' Transfer predicted poses to RTs '''
    RTs_all = np.ones((pred_pose.shape[0], pred_pose.shape[1], 4, 4))    #[bs, repeat_num, 4, 4]
    for i in range(pred_pose.shape[1]):
        R = get_rot_matrix(pred_pose[:, i, :-3], pose_mode)
        T = pred_pose[:, i, -3:]
        RTs = np.identity(4, dtype=float)[np.newaxis, ...].repeat(R.shape[0], 0)
        RTs[:, :3, :3] = R.cpu().numpy()
        RTs[:, :3, 3] = T.cpu().numpy()
        RTs_all[:, i, :, :] = RTs
    
    return RTs_all, pred_results


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
    return RTs_all, sorted_energy.cpu().numpy()


def inference_pose(data_path, inference_res_dir, pose_mode, record_process=False):
    ''' Create evaluation agent '''
    cfg.posenet_mode = 'score'
    score_agent = PoseNet(cfg)  
    
    ''' Load model '''
    score_agent.load_ckpt(model_dir=scorenet_ckpt_path, model_path=True, load_model_only=True)  
    
    ''' Load data '''
    detect_result, categorized_test_data = unpack_data(data_path)

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
            
    for key in categorized_test_data.keys():
        torch.cuda.empty_cache()
        num = len(categorized_test_data[key]['img_path'])
        index = [i * cfg.batch_size for i in range(0, num // cfg.batch_size + 1)]
        index = index if index[-1] == num else index + [num] 
        categorized_test_data[key]['pred_pose'] = []
        if record_process:
            categorized_test_data[key]['pred_pose_process'] = []
        
        print(f'Inferencing {key} category...')
        print(f'The num of instace is {num}!')
        
        ''' Start inference '''
        for i in tqdm(range(len(index) - 1)):
            t0 = time.time()   
            ''' Create batch data '''
            pts = categorized_test_data[key]['pts'][index[i]:index[i+1]]
            pts = torch.cuda.FloatTensor(np.array(pts))
            rgb = None if categorized_test_data[key]['rgb'] is None else \
                torch.cuda.FloatTensor(categorized_test_data[key]['rgb'][index[i]:index[i+1]])
            batch_sample = {'pts': pts}
            num_pts = pts.shape[1]
            zero_mean = torch.mean(pts[:, :, :3], dim=1)
            batch_sample['zero_mean_pts'] = copy.deepcopy(pts)
            batch_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
            batch_sample['pts_center'] = zero_mean
            ''' Predict poses '''
            video_save_path = f'{video_path[key]}/batch_{str(i)}'
            pred_RTs, pred_pose = pred_pose_batch(score_agent, batch_sample, video_save_path, pose_mode, record_process)     
            ''' Record results '''
            if record_process:
                categorized_test_data[key]['pred_pose'] += [pred_pose[0].cpu().numpy()[i] for i in range(pred_pose[0].shape[0])]
                categorized_test_data[key]['pred_pose_process'] += [pred_pose[1].cpu().numpy()[i] for i in range(pred_pose[1].shape[0])]
            else:
                categorized_test_data[key]['pred_pose'] += [pred_pose.cpu().numpy()[i] for i in range(pred_pose.shape[0])]
            img_path_list = categorized_test_data[key]['img_path'][index[i]:index[i+1]]
            inst_list = categorized_test_data[key]['inst'][index[i]:index[i+1]]
            for id, path in enumerate(img_path_list):
                detect_result[path]['result']['multi_hypothesis_pred_RTs'][inst_list[id]] = pred_RTs[id]           

    result_save_path = os.path.join(inference_res_dir, 'results_wo_energy.pkl')    
    cls_data_save_path = os.path.join(inference_res_dir, 'cls_data.pkl')
    with open(result_save_path, 'wb') as f:
        cPickle.dump(detect_result, f) 
    f.close()
    with open(cls_data_save_path, 'wb') as f:
        cPickle.dump(categorized_test_data, f) 
    f.close()
    

def inference_energy(inference_res_dir, pose_mode):
    ''' Create evaluation agent '''
    cfg.posenet_mode = 'energy'
    energy_agent = PoseNet(cfg)
    
    ''' Load model '''
    energy_agent.load_ckpt(model_dir=energynet_ckpt_path, model_path=True, load_model_only=True)  
    
    ''' Load data '''
    result_path = os.path.join(inference_res_dir, 'results_wo_energy.pkl')    
    cls_data_path = os.path.join(inference_res_dir, 'cls_data.pkl')
    with open(result_path, 'rb') as f:
        detect_result = cPickle.load(f)   
    f.close()
    with open(cls_data_path, 'rb') as f:
        categorized_test_data = cPickle.load(f)   
    f.close()
       
    for key in categorized_test_data.keys():
        torch.cuda.empty_cache()
        num = len(categorized_test_data[key]['img_path'])
        index = [i * cfg.batch_size for i in range(0, num // cfg.batch_size + 1)]
        index = index if index[-1] == num else index + [num] 
        print(f'Inferencing {key} category...')
        print(f'The num of instace is {num}!')
        
        # ''' Load model '''
        # eval_agent.load_ckpt(model_dir=ckpt_path_dict[key], model_path=True, load_model_only=True)
        
        ''' Start inference '''
        for i in tqdm(range(len(index) - 1)):   
            ''' Create batch data '''
            pts = categorized_test_data[key]['pts'][index[i]:index[i+1]]
            pts = torch.cuda.FloatTensor(np.array(pts))
            pred_pose = categorized_test_data[key]['pred_pose'][index[i]:index[i+1]]
            pred_pose = torch.cuda.FloatTensor(np.array(pred_pose))
            rgb = None if categorized_test_data[key]['rgb'] is None else \
                torch.cuda.FloatTensor(categorized_test_data[key]['rgb'][index[i]:index[i+1]])

            batch_sample = {'pts': pts}
            num_pts = pts.shape[1]
            zero_mean = torch.mean(pts[:, :, :3], dim=1)
            batch_sample['zero_mean_pts'] = copy.deepcopy(pts)
            batch_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
            batch_sample['pts_center'] = zero_mean
            
            ''' Predict energy '''
            sorted_pred_RTs, sorted_energy = pred_energy_batch(energy_agent, batch_sample, pred_pose, pose_mode)
            
            ''' Record results '''
            img_path_list = categorized_test_data[key]['img_path'][index[i]:index[i+1]]
            inst_list = categorized_test_data[key]['inst'][index[i]:index[i+1]]
            for id, path in enumerate(img_path_list):
                detect_result[path]['result']['multi_hypothesis_pred_RTs'][inst_list[id]] = sorted_pred_RTs[id]
                detect_result[path]['result']['energy'][inst_list[id]] = sorted_energy[id]
                if 'pred_pose_process' in categorized_test_data[key].keys():
                    detect_result[path]['result']['pred_pose_process'][inst_list[id]] = categorized_test_data[key]['pred_pose_process'][index[i]:index[i+1]][id]
            
    result_save_path = os.path.join(inference_res_dir, 'results_with_energy.pkl')    
    with open(result_save_path, 'wb') as f:
        cPickle.dump(detect_result, f) 
    f.close()

        
def evaluate(inference_res_dir, file_name):
    degree_thres_list = list(range(0, 46, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(inference_res_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    result_pkl_list = result_pkl_list
    assert len(result_pkl_list)
    
    with open(os.path.join(inference_res_dir, file_name), 'rb') as f:
        predictions = cPickle.load(f)
    f.close()
    
    pred_results = []
    for image_path in predictions.keys():
        # if not 'scene_3/0090' in image_path:
        #     continue
        result = predictions[image_path]['result']
        if 'gt_handle_visibility' not in result:
            result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
        else:
            assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                result['gt_handle_visibility'], result['gt_class_ids'])    
        # if 0 in result['gt_handle_visibility']:
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    
    save_path = {
        'inference_res_dir': f'{inference_res_dir}/results/{cfg.pooling_mode}/{cfg.ranker}',
        'ratio': 0,
        'pooling_mode': cfg.pooling_mode}
    exists_or_mkdir(save_path['inference_res_dir'])
    fw = open(f"{save_path['inference_res_dir']}/eval_logs.txt", 'a')
    fw.write(f"score_model: {cfg.score_model_dir}" + '\n')
    fw.write(f"energy_model: {cfg.energy_model_dir}" + '\n')
    fw.close()
    
    ratio_list = np.linspace(0.6, 0.6, 1)
    for ratio in ratio_list:
        print(f'pooling_mode: {cfg.pooling_mode}, ranker: {cfg.ranker}, ratio: {ratio}')
        # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation.(3D IoU -> 2D IoU)
        iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(
            pred_results, save_path['inference_res_dir'], degree_thres_list, shift_thres_list,
            iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True, 
            repeat_num=cfg.eval_repeat_num, pooling_mode=cfg.pooling_mode, 
            ratio=ratio, so3_vis=False, ranker=cfg.ranker)
        save_path['ratio'] = ratio
        record_results_and_draw_curves(save_path, iou_aps, pose_aps, iou_acc, pose_acc, iou_thres_list, degree_thres_list, shift_thres_list)


def detect_mrcnn_results(segmentation_results_path):
    if os.path.exists(segmentation_results_path):
        print('Segmentation results exist, skip detection! Load the segmentation results from {}'.format(segmentation_results_path))
    else:
        detect_mrcnn_genpose(segmentation_results_path)


def main():
    print('Detecting ...')
    detect_mrcnn_results(segmentation_results_path)
    print('Predict pose ...')
    inference_pose(segmentation_results_path, inference_res_dir, cfg.pose_mode, record_process=False)   
    print('Predict energy ...')
    inference_energy(inference_res_dir, cfg.pose_mode)
    print('Evaluating ...')
    evaluate(inference_res_dir, 'results_with_energy.pkl')


if __name__ == '__main__':
    main()
    
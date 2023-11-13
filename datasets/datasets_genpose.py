import sys
import os
import cv2
import random
import torch
import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import copy
import pytorch3d
sys.path.insert(0, '../')

from ipdb import set_trace
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from utils.data_augmentation import defor_2D, get_rotation
from utils.data_augmentation import data_augment
from utils.datasets_utils import aug_bbox_DZI, get_2d_coord_np, crop_resize_by_warp_affine
from utils.sgpa_utils import load_depth, get_bbox
from configs.config import get_config
from utils.misc import get_rot_matrix

    
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class NOCSDataSet(data.Dataset):
    def __init__(self, 
                 dynamic_zoom_in_params,
                 deform_2d_params,
                 source=None, 
                 mode='train', 
                 data_dir=None,
                 n_pts=1024, 
                 img_size=256, 
                 per_obj='',
                 ):
        '''
        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.dynamic_zoom_in_params = dynamic_zoom_in_params
        self.deform_2d_params = deform_2d_params

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        if source == 'CAMERA':
            self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = None
        
        # only train one object
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
                # iter over  all img_list, cal sublen

            if len(subset_len) == 2:
                camera_len  = 0
                real_len = 0
                for i in range(len(img_list_obj)):
                    if 'CAMERA' in img_list_obj[i].split('/'):
                        camera_len += 1
                    else:
                        real_len += 1
                self.subset_len = [camera_len, real_len]
            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(self.data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        # move the center to the body of the mug
        # meta info for re-label mug category
        with open(os.path.join(self.data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float32)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)

        self.invaild_list = []
        with open(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'), 'rb') as f:
            self.mug_sym = cPickle.load(f)
        # self.mug_sym = mmcv.load(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'))
        
        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):                
        img_path = os.path.join(self.data_dir, self.img_list[index])
        if img_path in self.invaild_list:
            return self.__getitem__((index + 1) % self.__len__())
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())
        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'

        # select one foreground object,
        # if specified, then select the object
        if self.per_obj != '':
            idx = gts['class_ids'].index(self.per_obj_id)
        else:
            idx = random.randint(0, len(gts['instance_ids']) - 1)
            ''' 
            ############### remove selected categories ###############
            remove_ids = self.cat_name2id['bowl']
            idx = None
            for i in range(10):
                idx_i = random.randint(0, len(gts['instance_ids']) - 1)
                if gts['class_ids'][idx_i] != remove_ids:
                    idx = idx_i
                    break
            if idx is None:
                return self.__getitem__((index + 1) % self.__len__())
            ##########################################################
            '''

        if gts['class_ids'][idx] == 6 and img_type == 'real':
            if self.mode == 'train':
                handle_tmp_path = img_path.split('/')
                scene_label = handle_tmp_path[-2] + '_res'
                img_id = int(handle_tmp_path[-1])
                mug_handle = self.mug_sym[scene_label][img_id]
            else:
                mug_handle = gts['handle_visibility'][idx]
        else:
            mug_handle = 1

        rgb = cv2.imread(img_path + '_color.png')
        if rgb is not None:
            rgb = rgb[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)

        depth_path = img_path + '_depth.png'
        
        if os.path.exists(depth_path):
            depth = load_depth(depth_path)
        else:
            return self.__getitem__((index + 1) % self.__len__())

        mask_path = img_path + '_mask.png'
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask[:, :, 2]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        # coord = cv2.imread(img_path + '_coord.png')
        # if coord is not None:
        #     coord = coord[:, :, :3]
        #     pass
        # else:
        #     return self.__getitem__((index + 1) % self.__len__())

        # aggragate information about the selected object
        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)
        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        mask_target = mask.copy().astype(np.float32)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0

        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )

        roi_depth = np.expand_dims(roi_depth, axis=0)
        # normalize depth
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())

        # cat_id, rotation translation and scale
        cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        # note that this is nocs model, normalized along diagonal axis
        model_name = gts['model_list'][idx]
        model = self.models[gts['model_list'][idx]].astype(np.float32)  # 1024 points
        nocs_scale = gts['scales'][idx]  # nocs_scale = image file / model file
        # fsnet scale (from model) scale residual
        fsnet_scale, mean_shape = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)
        fsnet_scale = fsnet_scale / 1000.0
        mean_shape = mean_shape / 1000.0
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]
        # add nnoise to roi_mask
        
        # pcl_in = self._depth_to_pcl(roi_depth, out_camK, roi_coord_2d, roi_mask) / 1000.0
        # np.savetxt('pts.txt', pcl_in)        
        roi_mask_def = defor_2D(
            roi_mask, 
            rand_r=self.deform_2d_params['roi_mask_r'], 
            rand_pro=self.deform_2d_params['roi_mask_pro']
        )
        pcl_in = self._depth_to_pcl(roi_depth, out_camK, roi_coord_2d, roi_mask_def) / 1000.0
        # np.savetxt('pts_def.txt', pcl_in)
        
        if len(pcl_in) < 50:
            return self.__getitem__((index + 1) % self.__len__())
        pcl_in = self._sample_points(pcl_in, self.n_pts)
        # sym
        sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)
        # generate augmentation parameters
        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()

        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        data_dict['cat_id'] = torch.as_tensor(cat_id, dtype=torch.int8).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info.astype(np.float32)).contiguous()
        data_dict['mean_shape'] = torch.as_tensor(mean_shape, dtype=torch.float32).contiguous()
        data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['model_point'] = torch.as_tensor(model, dtype=torch.float32).contiguous()
        data_dict['nocs_scale'] = torch.as_tensor(nocs_scale, dtype=torch.float32).contiguous()
        data_dict['handle_visibility'] = torch.as_tensor(int(mug_handle), dtype=torch.int8).contiguous()
        data_dict['path'] = img_path
        return data_dict


    def _get_depth_normalize(self, roi_depth, roi_m_d_valid):
        depth_v_value = roi_depth[roi_m_d_valid]
        depth_normalize = (roi_depth - np.min(depth_v_value)) / (np.max(depth_v_value) - np.min(depth_v_value))
        depth_normalize[~roi_m_d_valid] = 0.0
        return depth_normalize


    def _sample_points(self, pcl, n_pts):
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


    def _depth_to_pcl(self, depth, K, xymap, mask):
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
    
    
    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    def get_fs_net_scale(self, c, model, nocs_scale):
        # model pc x 3
        lx = max(model[:, 0]) - min(model[:, 0])
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale * 1000
        ly_t = ly * nocs_scale * 1000
        lz_t = lz * nocs_scale * 1000

        if c == 'bottle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'bowl':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'camera':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'can':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'laptop':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'mug':
            unitx = 146
            unity = 83
            unitz = 114
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
            raise NotImplementedError
        # scale residual
        return np.array([lx_t - unitx, ly_t - unity, lz_t - unitz]), np.array([unitx, unity, unitz])


    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int8)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int8)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int8)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int8)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int8)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=np.int8)  # for mug, we currently mark it as no symmetry
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=np.int8)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int8)
        return sym


def get_data_loaders(
    batch_size,
    seed,
    dynamic_zoom_in_params,
    deform_2d_params,
    percentage_data=1.0,
    data_path=None,
    source='CAMERA+Real',
    mode='train',
    n_pts=1024,
    img_size=256,
    per_obj='',
    num_workers=32,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = NOCSDataSet(
        dynamic_zoom_in_params=dynamic_zoom_in_params,
        deform_2d_params=deform_2d_params,
        source=source,
        mode=mode,
        data_dir=data_path,
        n_pts=n_pts,
        img_size=img_size,
        per_obj=per_obj,
    )
    
    if mode == 'train':
        shuffle = True
        num_workers = num_workers
    else:
        shuffle = False
        num_workers = 1

    if source == 'CAMERA+Real' and mode == 'train':
        # CAMERA : Real = 3 : 1
        camera_len = dataset.subset_len[0]
        real_len = dataset.subset_len[1]
        real_indices = list(range(camera_len, camera_len+real_len))
        camera_indices = list(range(camera_len))
        n_repeat = (camera_len // 3 - real_len) // real_len
        idx = camera_indices + real_indices*n_repeat
        random.shuffle(idx)
        size = int(percentage_data * len(idx))
        idx = idx[:size]
        data_sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=data_sampler,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True,
        )
        
    else:
        # sample
        size = int(percentage_data * len(dataset))
        dataset, _ = torch.utils.data.random_split(dataset, (size, len(dataset) - size))

        # train_dataloader = torch.utils.data.DataLoader(
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True,
        )
        
    return dataloader


def get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test']):
    data_loaders = {}
    if 'train' in data_type:
        train_loader = get_data_loaders(
            batch_size=cfg.batch_size, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_train,            
            data_path=cfg.data_path,
            source=cfg.train_source,
            mode='train',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['train_loader'] = train_loader
        
    if 'val' in data_type:
        val_loader = get_data_loaders(
            batch_size=cfg.mini_bs, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_val,            
            data_path=cfg.data_path,
            source=cfg.val_source,
            mode='test',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['val_loader'] = val_loader
        
    if 'test' in data_type:
        test_loader = get_data_loaders(
            batch_size=cfg.mini_bs, 
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_test,            
            data_path=cfg.data_path,
            source=cfg.test_source,
            mode='test',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['test_loader'] = test_loader
        
    return data_loaders


def process_batch(batch_sample,
                  device,
                  pose_mode='quat_wxyz',
                  mini_batch_size=None,
                  PTS_AUG_PARAMS=None):
    
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'], \
        f"the rotation mode {pose_mode} is not supported!"
    if PTS_AUG_PARAMS==None:
        PC_da = batch_sample['pcl_in'].to(device)
        gt_R_da = batch_sample['rotation'].to(device)
        gt_t_da = batch_sample['translation'].to(device)
    else:        
        PC_da, gt_R_da, gt_t_da, gt_s_da = data_augment(
            pts_aug_params=PTS_AUG_PARAMS,
            PC=batch_sample['pcl_in'].to(device), 
            gt_R=batch_sample['rotation'].to(device), 
            gt_t=batch_sample['translation'].to(device),
            gt_s=batch_sample['fsnet_scale'].to(device), 
            mean_shape=batch_sample['mean_shape'].to(device),
            sym=batch_sample['sym_info'].to(device),
            aug_bb=batch_sample['aug_bb'].to(device), 
            aug_rt_t=batch_sample['aug_rt_t'].to(device),
            aug_rt_r=batch_sample['aug_rt_R'].to(device),
            model_point=batch_sample['model_point'].to(device), 
            nocs_scale=batch_sample['nocs_scale'].to(device),
            obj_ids=batch_sample['cat_id'].to(device), 
        )

    processed_sample = {}
    processed_sample['pts'] = PC_da                # [bs, 1024, 3]
    processed_sample['pts_color'] = PC_da          # [bs, 1024, 3]
    processed_sample['id'] = batch_sample['cat_id'].to(device)      # [bs]
    processed_sample['handle_visibility'] = batch_sample['handle_visibility'].to(device)     # [bs]
    # processed_sample['path'] = batch_sample['path']
    if pose_mode == 'quat_xyzw':
        rot = pytorch3d.transforms.matrix_to_quaternion(gt_R_da)
    elif pose_mode == 'quat_wxyz':
        rot = pytorch3d.transforms.matrix_to_quaternion(gt_R_da)[:, [3, 0, 1, 2]]
    elif pose_mode == 'euler_xyz':
        rot = pytorch3d.transforms.matrix_to_euler_angles(gt_R_da, 'ZYX')
    elif pose_mode == 'euler_xyz_sx_cx':
        rot = pytorch3d.transforms.matrix_to_euler_angles(gt_R_da, 'ZYX')
        rot_sin_theta = torch.sin(rot)
        rot_cos_theta = torch.cos(rot)
        rot = torch.cat((rot_sin_theta, rot_cos_theta), dim=-1)
    elif pose_mode == 'rot_matrix':
        rot = pytorch3d.transforms.matrix_to_rotation_6d(gt_R_da.permute(0, 2, 1)).reshape(gt_R_da.shape[0], -1)
    else:
        raise NotImplementedError
    
    location = gt_t_da # [bs, 3]
    processed_sample['gt_pose'] = torch.cat([rot.float(), location.float()], dim=-1)   # [bs, 4/6/3 + 3]
    
    """ zero center """
    num_pts = processed_sample['pts'].shape[1]
    zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)
    processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
    processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
    processed_sample['zero_mean_gt_pose'] = copy.deepcopy(processed_sample['gt_pose'])
    processed_sample['zero_mean_gt_pose'][:, -3:] -= zero_mean
    processed_sample['pts_center'] = zero_mean
    
    if 'color' in batch_sample.keys():
        pass
        # processed_sample['color'] = batch_sample['color'].to(device)       # [bs]

    if not mini_batch_size == None:
        for key in processed_sample.keys():
            processed_sample[key] = processed_sample[key][:mini_batch_size]
    
    if not 'color' in processed_sample.keys():
        pass
        # processed_sample['color'] = None
    # print(processed_sample['zero_mean_pts'].device)
    return processed_sample 
    

if __name__ == '__main__':
    cfg = get_config()
    cfg.pose_mode = 'rot_matrix'
    data_loaders = get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test'])
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    test_loader = data_loaders['test_loader']
    for index, batch_sample in enumerate(tqdm(test_loader)):
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
    for index, batch_sample in enumerate(tqdm(val_loader)):
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
    for index, batch_sample in enumerate(tqdm(train_loader)):
        batch_sample = process_batch(
            batch_sample = batch_sample, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )
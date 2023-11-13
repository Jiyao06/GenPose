#   introduced from fs-net
import numpy as np
import cv2
import torch
import math
from utils.visualize import show_point_cloud


# add noise to mask
def defor_2D(roi_mask, rand_r=2, rand_pro=0.3):
    '''

    :param roi_mask: 256 x 256
    :param rand_r: randomly expand or shrink the mask iter rand_r
    :return:
    '''
    roi_mask = roi_mask.copy().squeeze()
    if np.random.rand() > rand_pro:
        return roi_mask
    mask = roi_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_erode = cv2.erode(mask, kernel, rand_r)  # rand_r
    mask_dilate = cv2.dilate(mask, kernel, rand_r)
    change_list = roi_mask[mask_erode != mask_dilate]
    l_list = change_list.size
    if l_list < 1.0:
        return roi_mask
    choose = np.random.choice(l_list, l_list // 2, replace=False)
    change_list = np.ones_like(change_list)
    change_list[choose] = 0.0
    roi_mask[mask_erode != mask_dilate] = change_list
    roi_mask[roi_mask > 0.0] = 1.0
    return roi_mask


# point cloud based data augmentation
# augment based on bounding box
def defor_3D_bb(pc, R, t, s, sym=None, aug_bb=None):
    # pc  n x 3, here s must  be the original s
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    if sym[0] == 1:  # y axis symmetry
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        exz = (ex + ez) / 2
        pc_reproj[:, (0, 2)] = pc_reproj[:, (0, 2)] * exz
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        s[0] = s[0] * exz
        s[1] = s[1] * ey
        s[2] = s[2] * exz
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        return pc_new, s
    else:
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        pc_reproj[:, 0] = pc_reproj[:, 0] * ex
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        pc_reproj[:, 2] = pc_reproj[:, 2] * ez
        s[0] = s[0] * ex
        s[1] = s[1] * ey
        s[2] = s[2] * ez
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        return pc_new, s


def defor_3D_bb_in_batch(pc, model_point, R, t, s, sym=None, aug_bb=None):
    pc_reproj = torch.matmul(R.transpose(-1, -2), (pc - t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)
    sym_aug_bb = (aug_bb + aug_bb[:, [2, 1, 0]]) / 2.0
    sym_flag = (sym[:, 0] == 1).unsqueeze(-1)
    new_aug_bb = torch.where(sym_flag, sym_aug_bb, aug_bb)
    pc_reproj = pc_reproj * new_aug_bb.unsqueeze(-2)
    model_point_new = model_point * new_aug_bb.unsqueeze(-2)
    pc_new = (torch.matmul(R, pc_reproj.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)
    s_new = s * new_aug_bb
    return pc_new, s_new, model_point_new


def defor_3D_bc(pc, R, t, s, model_point, nocs_scale):
    # resize box cage along y axis, the size s is modified
    ey_up = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand(1,  device=pc.device) * (1.2 - 0.8) + 0.8
    # for each point, resize its x and z linealy
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = (pc_reproj[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    model_point_resize = (model_point[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = max(model_point[:, 0]) - min(model_point[:, 0])
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * nocs_scale
    ly_t = ly * nocs_scale
    lz_t = lz * nocs_scale
    return pc_new, torch.tensor([lx_t, ly_t, lz_t], device=pc.device)


def defor_3D_bc_in_batch(pc, R, t, s, model_point, nocs_scale):
    # resize box cage along y axis, the size s is modified
    bs = pc.size(0)
    ey_up = torch.rand((bs,1), device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand((bs, 1),  device=pc.device) * (1.2 - 0.8) + 0.8
    pc_reproj = torch.matmul(R.transpose(-1,-2), (pc-t.unsqueeze(-2)).transpose(-1,-2)).transpose(-1,-2)

    s_y = s[..., 1].unsqueeze(-1)
    per_point_resize = (pc_reproj[..., 1] + s_y / 2.0) / s_y * (ey_up - ey_down) + ey_down
    pc_reproj[..., 0] = pc_reproj[..., 0] * per_point_resize
    pc_reproj[..., 2] = pc_reproj[..., 2] * per_point_resize
    pc_new = (torch.matmul(R, pc_reproj.transpose(-2,-1)) + t.unsqueeze(-1)).transpose(-2,-1)


    new_model_point = model_point*1.0
    model_point_resize = (new_model_point[..., 1] + s_y / 2) / s_y * (ey_up - ey_down) + ey_down
    new_model_point[..., 0] = new_model_point[..., 0] * model_point_resize
    new_model_point[..., 2] = new_model_point[..., 2] * model_point_resize

    s_new = (torch.max(new_model_point, dim=1)[0] - torch.min(new_model_point, dim=1)[0])*nocs_scale.unsqueeze(-1)
    return pc_new, s_new, ey_up, ey_down

# def defor_3D_pc(pc, r=0.05):
#     points_defor = torch.randn(pc.shape).to(pc.device)
#     pc = pc + points_defor * r * pc
#     return pc

def defor_3D_pc(pc, gt_t, r=0.2, points_defor=None, return_defor=False):

    if points_defor is None:
        points_defor = torch.rand(pc.shape).to(pc.device)*r
    new_pc = pc + points_defor*(pc-gt_t.unsqueeze(1))
    if return_defor:
        return new_pc, points_defor
    return new_pc


# point cloud based data augmentation
# random rotation and translation
def defor_3D_rt(pc, R, t, aug_rt_t, aug_rt_r):
    #  add_t
    dx = aug_rt_t[0]
    dy = aug_rt_t[1]
    dz = aug_rt_t[2]

    pc[:, 0] = pc[:, 0] + dx
    pc[:, 1] = pc[:, 1] + dy
    pc[:, 2] = pc[:, 2] + dz
    t[0] = t[0] + dx
    t[1] = t[1] + dy
    t[2] = t[2] + dz

    # add r
    '''
    Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    Rm_tensor = torch.tensor(Rm, device=pc.device)
    pc_new = torch.mm(Rm_tensor, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm_tensor, R)
    R = R_new
    '''
    '''
    x_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    y_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    z_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    Rm = get_rotation_torch(x_rot, y_rot, z_rot)
    '''
    Rm = aug_rt_r
    pc_new = torch.mm(Rm, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm, R)
    R = R_new
    T_new = torch.mm(Rm, t.view(3, 1))
    t = T_new

    return pc, R, t


def defor_3D_rt_in_batch(pc, R, t, aug_rt_t, aug_rt_r):
    pc_new = pc + aug_rt_t.unsqueeze(-2)
    t_new = t + aug_rt_t
    pc_new = torch.matmul(aug_rt_r, pc_new.transpose(-2,-1)).transpose(-2,-1)

    R_new = torch.matmul(aug_rt_r, R)
    t_new = torch.matmul(aug_rt_r, t_new.unsqueeze(-1)).squeeze(-1)
    return pc_new, R_new, t_new


def get_rotation(x_, y_, z_):
    # print(math.cos(math.pi/2))
    x = float(x_ / 180) * math.pi
    y = float(y_ / 180) * math.pi
    z = float(z_ / 180) * math.pi
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x)).astype(np.float32)


def get_rotation_torch(x_, y_, z_):
    x = (x_ / 180) * math.pi
    y = (y_ / 180) * math.pi
    z = (z_ / 180) * math.pi
    R_x = torch.tensor([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]], device=x_.device)

    R_y = torch.tensor([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]], device=y_.device)

    R_z = torch.tensor([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]], device=z_.device)
    return torch.mm(R_z, torch.mm(R_y, R_x))


def data_augment(pts_aug_params, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                        model_point, nocs_scale, obj_ids, check_points=False):
    """
    PC torch.Size([32, 1028, 3])
    gt_R torch.Size([32, 3, 3])
    gt_t torch.Size([32, 3])
    gt_s torch.Size([32, 3])
    mean_shape torch.Size([32, 3])
    sym torch.Size([32, 4])
    aug_bb torch.Size([32, 3])
    aug_rt_t torch.Size([32, 3])
    aug_rt_r torch.Size([32, 3, 3])
    model_point torch.Size([32, 1024, 3])
    nocs_scale torch.Size([32])
    obj_ids torch.Size([32])
    """

    def aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag):
        PC_new, gt_s_new, model_point_new = defor_3D_bb_in_batch(PC, model_point, gt_R, gt_t, gt_s + mean_shape, sym, aug_bb)
        gt_s_new = gt_s_new - mean_shape
        PC = torch.where(flag.unsqueeze(-1), PC_new, PC)
        gt_s = torch.where(flag, gt_s_new, gt_s)
        model_point_new = torch.where(flag.unsqueeze(-1), model_point_new, model_point)
        return PC, gt_s, model_point_new

    def aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag):
        PC_new, gt_R_new, gt_t_new = defor_3D_rt_in_batch(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
        PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
        gt_R_new = torch.where(flag.unsqueeze(-1), gt_R_new, gt_R)
        gt_t_new = torch.where(flag, gt_t_new, gt_t)
        return PC_new, gt_R_new, gt_t_new

    def aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag):
        pc_new, s_new, ey_up, ey_down = defor_3D_bc_in_batch(PC, gt_R, gt_t, gt_s + mean_shape, model_point,
                                                                nocs_scale)
        pc_new = torch.where(flag.unsqueeze(-1), pc_new, PC)
        s_new = torch.where(flag, s_new - mean_shape, gt_s)
        return pc_new, s_new, ey_up, ey_down

    def aug_pc_with_flag(PC, gt_t, flag, aug_pc_r):
        PC_new, defor = defor_3D_pc(PC, gt_t, aug_pc_r, return_defor=True)
        PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
        return PC_new, defor
    

    # augmentation
    bs = PC.shape[0]

    prob_bb = torch.rand((bs, 1), device=PC.device)
    flag = prob_bb < pts_aug_params['aug_bb_pro']
    PC, gt_s, model_point = aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag)

    prob_rt = torch.rand((bs, 1), device=PC.device)
    flag = prob_rt < pts_aug_params['aug_rt_pro']
    PC, gt_R, gt_t = aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag)

    # only do bc for mug and bowl
    prob_bc = torch.rand((bs, 1), device=PC.device)
    flag = torch.logical_and(prob_bc < pts_aug_params['aug_bc_pro'], torch.logical_or(obj_ids== 5, obj_ids == 1).unsqueeze(-1))
    PC, gt_s, _, _ = aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag)

    prob_pc = torch.rand((bs, 1), device=PC.device)
    flag = prob_pc < pts_aug_params['aug_pc_pro']
    PC, _ = aug_pc_with_flag(PC, gt_t, flag, pts_aug_params['aug_pc_r'])

    if check_points:
        pc_reproj = torch.matmul(gt_R.transpose(-1, -2), (PC - gt_t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)
        model_point *= nocs_scale.unsqueeze(-1).unsqueeze(-1)
        for i in range(len(pc_reproj)):
            show_point_cloud([pc_reproj[i].detach().cpu().numpy(), model_point[i].detach().cpu().numpy()], colors=[[0,0,1], [1,0,0]], axis_size=0.1)

    return PC, gt_R, gt_t, gt_s

import numpy as np
import os
import copy
import pytorch3d
import pytorch3d.io
import torch
import torch.distributed as dist
from ipdb import set_trace
os.sys.path.append('..')
from utils.genpose_utils import get_pose_dim
from scipy.spatial.transform import Rotation as R


def parallel_setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(seed)


def parallel_cleanup():
    dist.destroy_process_group()
    

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def depth2xyz(depth_img, camera_params):
    # scale camera parameters
    h, w = depth_img.shape
    scale_x = w / camera_params['xres']
    scale_y = h / camera_params['yres']
    fx = camera_params['fx'] * scale_x
    fy = camera_params['fy'] * scale_y
    x_offset = camera_params['cx'] * scale_x
    y_offset = camera_params['cy'] * scale_y

    indices = np.indices((h, w), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def fps_down_sample(vertices, num_point_sampled):
    # FPS down sample 
    # vertices.shape = (N,3) or (N,2)
    
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0)          # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d)                # Select the point farthest from the center of gravity as the starting point
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_)        # Whether the point is selected
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    return vertices[flags]


def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        # print(N)
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)
    

def trans_form_quat_and_location(quaternion, location, quat_type='wxyz'):
    assert quat_type in ['wxyz', 'xyzw'], f"The type of quaternion {quat_type} is not supported!"
    
    if quat_type == 'xyzw':
        quaternion_xyzw = quaternion
    else:
        quaternion_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        
    scipy_rot = R.from_quat(quaternion_xyzw)
    rot = scipy_rot.as_matrix()
    
    location = location[np.newaxis, :].T
    transformation = np.concatenate((rot, location), axis=1)
    transformation = np.concatenate((transformation, np.array([[0, 0, 0, 1]])), axis=0)
    return transformation
   
   
def get_rot_matrix(batch_pose, pose_mode='quat_wxyz'):
    """
    pose_mode: 
        'quat_wxyz'  -> batch_pose [B, 4]
        'quat_xyzw'  -> batch_pose [B, 4] 
        'euler_xyz'  -> batch_pose [B, 3] 
        'rot_matrix' -> batch_pose [B, 6]
        
    Return: rot_matrix [B, 3, 3]
    """
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"
        
    if pose_mode in ['quat_wxyz', 'quat_xyzw']:
        if pose_mode == 'quat_wxyz':
            quat_wxyz = batch_pose
        else:
            index = [3, 0, 1, 2]
            quat_wxyz = batch_pose[:, index]
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(quat_wxyz)
            
    elif pose_mode == 'rot_matrix':
        rot_mat= pytorch3d.transforms.rotation_6d_to_matrix(batch_pose).permute(0, 2, 1)
        
    elif pose_mode == 'euler_xyz_sx_cx':
        rot_sin_theta = batch_pose[:, :3]
        rot_cos_theta = batch_pose[:, 3:6]
        theta = torch.atan2(rot_sin_theta, rot_cos_theta)
        rot_mat = pytorch3d.transforms.euler_angles_to_matrix(theta, 'ZYX')
    elif pose_mode == 'euler_xyz':
        rot_mat = pytorch3d.transforms.euler_angles_to_matrix(batch_pose, 'ZYX')
    else:
        raise NotImplementedError
    
    return rot_mat

  
def transform_single_pts(pts, transformation):
    N = pts.shape[0]
    pts = np.concatenate((pts.T, np.ones(N)[np.newaxis, :]), axis=0)
    new_pts = transformation @ pts
    return new_pts.T[:, :3]


def transform_batch_pts(batch_pts, batch_pose, pose_mode='quat_wxyz', inverse_pose=False):
    """
    Args:
        batch_pts [B, N, C], N is the number of points, and C [x, y, z, ...]
        batch_pose [B, C], [quat/rot_mat/euler, trans]
        pose_mode is from ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'rot_matrix']
        if inverse_pose is true, the transformation will be inversed
    Returns:
        new_pts [B, N, C]
    """
    assert pose_mode in ['quat_wxyz', 'quat_xyzw', 'euler_xyz', 'euler_xyz_sx_cx', 'rot_matrix'],\
        f"the rotation mode {pose_mode} is not supported!"
        
    B = batch_pts.shape[0]
    index = get_pose_dim(pose_mode) - 3
    rot = batch_pose[:, :index]
    loc = batch_pose[:, index:]

    rot_mat = get_rot_matrix(rot, pose_mode)
    if inverse_pose == True:
        rot_mat, loc = inverse_RT(rot_mat, loc)
    loc = loc[..., np.newaxis]    
    
    trans_mat = torch.cat((rot_mat, loc), dim=2)
    trans_mat = torch.cat((trans_mat, torch.tile(torch.tensor([[0, 0, 0, 1]]).to(trans_mat.device), (B, 1, 1))), dim=1)
    
    new_pts = copy.deepcopy(batch_pts)
    padding = torch.ones([batch_pts.shape[0], batch_pts.shape[1], 1]).to(batch_pts.device)
    pts = torch.cat((batch_pts[:, :, :3], padding), dim=2) 
    new_pts[:, :, :3] = torch.matmul(trans_mat.to(torch.float32), pts.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
    
    return new_pts
  
  
def inverse_RT(batch_rot_mat, batch_trans):
    """
    Args: 
        batch_rot_mat [B, 3, 3]
        batch_trans [B, 3]
    Return:
        inversed_rot_mat [B, 3, 3]
        inversed_trans [B, 3]       
    """
    trans = batch_trans[..., np.newaxis]
    inversed_rot_mat = batch_rot_mat.permute(0, 2, 1)
    inversed_trans = - inversed_rot_mat @ trans
    return inversed_rot_mat, inversed_trans.squeeze(-1)


""" https://arc.aiaa.org/doi/abs/10.2514/1.28949 """
""" https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions """
""" http://tbirdal.blogspot.com/2019/10/i-allocate-this-post-to-providing.html """
def average_quaternion_torch(Q, weights=None):
    if weights is None:
        weights = torch.ones(len(Q), device=Q.device) / len(Q)
    A = torch.zeros((4, 4), device=Q.device)
    weight_sum = torch.sum(weights)

    oriented_Q = ((Q[:, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("bi,bk->bik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("bij,b->bij", (A, weights)), 0)
    A /= weight_sum

    q_avg = torch.linalg.eigh(A)[1][:, -1]
    if q_avg[0] < 0:
        return -q_avg
    return q_avg


def average_quaternion_batch(Q, weights=None):
    """calculate the average quaternion of the multiple quaternions
    Args:
        Q (tensor): [B, num_quaternions, 4]
        weights (tensor, optional): [B, num_quaternions]. Defaults to None.

    Returns:
        oriented_q_avg: average quaternion, [B, 4]
    """
    
    if weights is None:
        weights = torch.ones((Q.shape[0], Q.shape[1]), device=Q.device) / Q.shape[1]
    A = torch.zeros((Q.shape[0], 4, 4), device=Q.device)
    weight_sum = torch.sum(weights, axis=-1)

    oriented_Q = ((Q[:, :, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("abi,abk->abik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("abij,ab->abij", (A, weights)), 1)
    A /= weight_sum.reshape(A.shape[0], -1).unsqueeze(-1).repeat(1, 4, 4)

    q_avg = torch.linalg.eigh(A)[1][:, :, -1]
    oriented_q_avg = ((q_avg[:, 0:1] > 0).float() - 0.5) * 2 * q_avg
    return oriented_q_avg


def average_quaternion_numpy(Q, W=None):
    if W is not None:
        Q *= W[:, None]
    eigvals, eigvecs = np.linalg.eig(Q.T@Q)
    return eigvecs[:, eigvals.argmax()]


def normalize_rotation(rotation, rotation_mode):
    if rotation_mode == 'quat_wxyz' or rotation_mode == 'quat_xyzw':
        rotation /= torch.norm(rotation, dim=-1, keepdim=True)
    elif rotation_mode == 'rot_matrix':
        rot_matrix = get_rot_matrix(rotation, rotation_mode)
        rotation[:, :3] = rot_matrix[:, :, 0]
        rotation[:, 3:6] = rot_matrix[:, :, 1]
    elif rotation_mode == 'euler_xyz_sx_cx':
        rot_sin_theta = rotation[:, :3]
        rot_cos_theta = rotation[:, 3:6]
        theta = torch.atan2(rot_sin_theta, rot_cos_theta)
        rotation[:, :3] = torch.sin(theta)
        rotation[:, 3:6] = torch.cos(theta)
    elif rotation_mode == 'euler_xyz':
        pass
    else:
        raise NotImplementedError
    return rotation

    
if __name__ == '__main__':
    quat = torch.randn(2, 3, 4)
    quat = quat / torch.linalg.norm(quat, axis=-1).unsqueeze(-1)
    quat = average_quaternion_batch(quat)
    

    
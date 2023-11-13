''' modified from CAPTRA https://github.com/HalfSummer11/CAPTRA/tree/5d7d088c3de49389a90b5fae280e96409e7246c6 '''

import torch
import copy
import math
from ipdb import set_trace

def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(norm)


def matrix_to_unit_quaternion(matrix):
    assert matrix.shape[-1] == matrix.shape[-2] == 3
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    trace = 1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    trace = torch.clamp(trace, min=0.)
    r = torch.sqrt(trace)
    s = 1.0 / (2 * r + 1e-7)
    w = 0.5 * r
    x = (matrix[..., 2, 1] - matrix[..., 1, 2])*s
    y = (matrix[..., 0, 2] - matrix[..., 2, 0])*s
    z = (matrix[..., 1, 0] - matrix[..., 0, 1])*s

    q = torch.stack((w, x, y, z), dim=-1)

    return normalize(q)


def generate_random_quaternion(quaternion_shape):
    assert quaternion_shape[-1] == 4
    rand_norm = torch.randn(quaternion_shape)
    rand_q = normalize(rand_norm)
    return rand_q


def jitter_quaternion(q, theta):  #[Bs, 4], [Bs, 1]
    new_q = generate_random_quaternion(q.shape).to(q.device)
    dot_product = torch.sum(q*new_q, dim=-1, keepdim=True)  #
    shape = (tuple(1 for _ in range(len(dot_product.shape) - 1)) + (4, ))
    q_orthogonal = normalize(new_q - q * dot_product.repeat(*shape))
    # theta = 2arccos(|p.dot(q)|)
    # |p.dot(q)| = cos(theta/2)
    tile_theta = theta.repeat(shape)
    jittered_q = q*torch.cos(tile_theta/2) + q_orthogonal*torch.sin(tile_theta/2)

    return jittered_q


def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1)
    norm_check =  (norm - 1.0).abs()
    try:
        assert torch.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(torch.max(norm_check)))
        return -1
    return 0


def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z= torch.unbind(q, dim=-1)
    matrix = torch.stack(( 1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y* w,
                        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,  
                        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x -2*y*y),
                        dim=-1)
    matrix_shape = list(matrix.shape)[:-1]+[3,3]
    return matrix.view(matrix_shape).contiguous()


def noisy_rot_matrix(matrix, rad, type='normal'):
    if type == 'normal':
        theta = torch.abs(torch.randn_like(matrix[..., 0, 0])) * rad
    elif type == 'uniform':
        theta = torch.rand_like(matrix[..., 0, 0]) * rad
    quater = matrix_to_unit_quaternion(matrix)
    new_quater = jitter_quaternion(quater, theta.unsqueeze(-1))
    new_mat = unit_quaternion_to_matrix(new_quater)
    return new_mat


def add_noise_to_RT(RT, type='normal', r=5.0, t=0.03):
    rand_type = type  # 'uniform' or 'normal' --> we use 'normal'

    def random_tensor(base):
        if rand_type == 'uniform':
            return torch.rand_like(base) * 2.0 - 1.0
        elif rand_type == 'normal':
            return torch.randn_like(base)
    new_RT = copy.deepcopy(RT)
    new_RT[:, :3, :3] = noisy_rot_matrix(RT[:, :3, :3], r/180*math.pi, type=rand_type).reshape(RT[:, :3, :3].shape)
    norm = random_tensor(RT[:, 0, 0]) * t  # [B, P]
    direction = random_tensor(RT[:, :3, 3].squeeze(-1))  # [B, P, 3]
    direction = direction / torch.clamp(direction.norm(dim=-1, keepdim=True), min=1e-9)  # [B, P, 3] unit vecs
    new_RT[:, :3, 3] = RT[:, :3, 3] + (direction * norm.unsqueeze(-1))  # [B, P, 3, 1]

    return new_RT


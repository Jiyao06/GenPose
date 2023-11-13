"""
    Evaluation-related codes are modified from
    https://github.com/hughw19/NOCS_CVPR2019
"""
import logging
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import pytorch3d
import pytorch3d.io
import torch
import copy

from tqdm import tqdm
from ipdb import set_trace

from utils.misc import average_quaternion_batch
from utils.visualize import so3_visualization


def sort_sRT(poses, energy):
    """  Rank the poses from highest to lowest energy 
    
    Args:
        poses (np.array): [bs, inference_num, 4, 4]
        energy (np.array): [bs, inference_num, 2]
        
    Returns:
        sorted_poses (np.array): [bs, inference_num, pose_dim]
        sorted_energy (np.array): [bs, inference_num, 2]        
    """
    # get the sorted energy 
    bs = poses.shape[0]
    repeat_num= poses.shape[1]
    sorted_energy = -np.sort(-energy, axis=1)
    indices_1 = np.argsort(-energy, axis=1)
    indices_0 = np.arange(0, bs).reshape(1, -1).repeat(repeat_num, axis=0).reshape(1, -1)
    
    indices_1_rot = indices_1.transpose(2, 1, 0)[0].reshape(1, -1)
    indices_1_trans = indices_1.transpose(2, 1, 0)[1].reshape(1, -1)
    
    rot_index = np.concatenate((indices_0, indices_1_rot), axis=0).tolist()
    trans_index = np.concatenate((indices_0, indices_1_trans), axis=0).tolist()
    sorted_poses = poses[rot_index[0], rot_index[1]]
    sorted_poses[:, :, 3] = poses[trans_index[0], trans_index[1]][:, :, 3]
    sorted_poses = sorted_poses.reshape(repeat_num, bs, 4, 4).transpose(1, 0, 2, 3)
    
    return sorted_poses, sorted_energy


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def load_obj(path_to_file):
    """ Load obj file.

    Args:
        path_to_file: path

    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices

    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces


def create_sphere():
    # 642 verts, 1280 faces,
    verts, faces = load_obj('assets/sphere_mesh_template.obj')
    return verts, faces


def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
        sqrt_r1 * r2 * face_vertices[2, :]

    return point


def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C


def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points


def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts


def sample_points_from_mesh(path, n_pts, with_normal=False, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True

    """
    vertices, faces = load_obj(path)
    if fps:
        points = uniform_sample(vertices, faces, ratio*n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal)
    return points


def load_depth(img_path):
    """ Load depth image from img_path. """
    if img_path[-4:] == '.png':
        depth_path = img_path
    else:
        depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def compute_sRT_errors(sRT1, sRT2):
    """
    Args:
        sRT1: [4, 4]. homogeneous affine transformation
        sRT2: [4, 4]. homogeneous affine transformation

    Returns:
        R_error: angle difference in degree,
        T_error: Euclidean distance
        IoU: relative scale error

    """
    try:
        assert np.array_equal(sRT1[3, :], sRT2[3, :])
        assert np.array_equal(sRT1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT1[3, :], sRT2[3, :])

    s1 = np.cbrt(np.linalg.det(sRT1[:3, :3]))
    R1 = sRT1[:3, :3] / s1
    T1 = sRT1[:3, 3]
    s2 = np.cbrt(np.linalg.det(sRT2[:3, :3]))
    R2 = sRT2[:3, :3] / s2
    T2 = sRT2[:3, 3]
    R12 = R1 @ R2.transpose()
    R_error = np.arccos(np.clip((np.trace(R12)-1)/2, -1.0, 1.0)) * 180 / np.pi
    T_error = np.linalg.norm(T1 - T2)
    IoU = np.abs(s1 - s2) / s2

    return R_error, T_error, IoU


############################################################
#  Evaluation
############################################################

def get_3d_coord_axis():
    return np.array(
        [[0, 0, 0], 
         [0, 0, 0.1], 
         [0, 0.1, 0], 
         [0.1, 0, 0]]
        ).transpose()


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2, handle_visibility):
    """ Computes IoU overlaps between two 3D bboxes. """
    def asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2):
        noc_cube_1 = get_3d_bbox(size_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, sRT_1)
        noc_cube_2 = get_3d_bbox(size_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, sRT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if sRT_1 is None or sRT_2 is None:
        return -1

    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or \
        (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):
        def y_rotation_matrix(theta):
            return np.array([[ np.cos(theta), 0, np.sin(theta), 0],
                             [ 0,             1, 0,             0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [ 0,             0, 0,             1]])
        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = sRT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, sRT_2, size_1, size_2))
    else:
        max_iou = asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2)

    return max_iou


def compute_IoU_matches(gt_class_ids, gt_sRT, gt_size, gt_handle_visibility,
                        pred_class_ids, pred_sRT, pred_size, pred_scores,
                        synset_names, iou_3d_thresholds, score_threshold=0):
    """ Find matches between NOCS prediction and ground truth instances.

    Args:
        size: 3D bounding box size
        bboxes: 2D bounding boxes

    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of the matched ground truth box.
        overlaps: IoU overlaps.
        indices:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    if num_pred:
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_class_ids = pred_class_ids[indices].copy()
        pred_size = pred_size[indices].copy()
        pred_sRT = pred_sRT[indices].copy()
    # compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_IoU(pred_sRT[i], gt_sRT[j], pred_size[i, :], gt_size[j],
                synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]], gt_handle_visibility[j])
    # loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])
    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(indices.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue
                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break
    return gt_matches, pred_matches, overlaps, indices


def compute_2d_IoU(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        bbox1 (list): coordinates of first bbox [x1, y1, x2, y2]
        bbox2 (list): coordinates of second bbox [x1, y1, x2, y2]

    Returns:
        float: Intersection over Union (IoU) value
    """
    # Calculate intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of bboxes
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area

    return iou


def compute_2d_IoU_matches(gt_class_ids, gt_bboxes,
                           pred_class_ids, pred_bboxes, pred_scores, 
                           iou_2d_thresholds, score_threshold=0):
    """ Find matches between MaskRCNN prediction and ground truth instances.

    Args:
        size: 3D bounding box size
        bboxes: 2D bounding boxes

    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of the matched ground truth box.
        overlaps: IoU overlaps.
        indices:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    if num_pred:
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_class_ids = pred_class_ids[indices].copy()
    # compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_2d_IoU(pred_bboxes[i], gt_bboxes[j])
    # loop through predictions and find matching ground truth boxes
    num_iou_2d_thres = len(iou_2d_thresholds)
    pred_matches = -1 * np.ones([num_iou_2d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_2d_thres, num_gt])
    for s, iou_thres in enumerate(iou_2d_thresholds):
        for i in range(indices.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue
                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break
    return gt_matches, pred_matches, overlaps, indices


def compute_RT_errors(sRT_1, sRT_2, class_id, handle_visibility, synset_names):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal( sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
        (synset_names[class_id] == 'mug' and handle_visibility == 0):
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    else:
        R = R1 @ R2.transpose()
        cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def compute_RT_overlaps(gt_class_ids, gt_sRT, gt_handle_visibility, pred_class_ids, pred_sRT, synset_names):
    """ Finds overlaps between prediction and ground truth instances.

    Returns:
        overlaps:

    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2))
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_errors(pred_sRT[i], gt_sRT[j], gt_class_ids[j],
                                                  gt_handle_visibility[j], synset_names)
    return overlaps


def compute_RT_matches(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue
                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_ap_and_acc(pred_matches, pred_scores, gt_matches):
    # sort the scores from high to low
    assert pred_matches.shape[0] == pred_scores.shape[0]
    score_indices = np.argsort(pred_scores)[::-1]
    # pred_scores = pred_scores[score_indices]
    pred_matches = pred_matches[score_indices]
    precisions = np.cumsum(pred_matches > -1) / (np.arange(len(pred_matches)) + 1)
    recalls = np.cumsum(pred_matches > -1).astype(np.float32) / len(gt_matches)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    # compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    # accuracy
    acc = np.sum(pred_matches > -1) / len(pred_matches)

    return ap, acc


'''
def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """ Compute mean Average Precision.

    Returns:
        iou_aps:
        pose_aps:
        iou_acc:
        pose_acc:
    """
    
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)
    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    # pre-allocate more than enough memory
    iou_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_acc = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_count = [0 for _ in range(num_classes)]
    iou_gt_count = [0 for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_acc = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_gt_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_count = [0 for _ in range(num_classes)]
    pose_gt_count = [0 for _ in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(tqdm(pred_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_sRT = np.array(result['gt_RTs'])
        gt_size = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_class_ids = result['pred_class_ids']
        pred_sRT = np.array(result['pred_RTs'])
        pred_size = result['pred_scales']
        pred_scores = result['pred_scores']

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue
        
        set_trace()
        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_sRT = gt_sRT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
            cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)

            # calculate the overlap between each gt instance and pred instance
            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = \
                compute_IoU_matches(cls_gt_class_ids, cls_gt_sRT, cls_gt_size, cls_gt_handle_visibility,
                                    cls_pred_class_ids, cls_pred_sRT, cls_pred_size, cls_pred_scores,
                                    synset_names, iou_thres_list)
            set_trace()
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
            
            num_pred = iou_cls_pred_match.shape[1]
            pred_start = iou_pred_count[cls_id]
            pred_end = pred_start + num_pred
            iou_pred_count[cls_id] = pred_end
            iou_pred_matches_all[cls_id][:, pred_start:pred_end] = iou_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            assert cls_pred_scores_tile.shape[1] == num_pred
            iou_pred_scores_all[cls_id][:, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = iou_cls_gt_match.shape[1]
            gt_start = iou_gt_count[cls_id]
            gt_end = gt_start + num_gt
            iou_gt_count[cls_id] = gt_end
            iou_gt_matches_all[cls_id][:, gt_start:gt_end] = iou_cls_gt_match
            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)
                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_sRT, synset_names)
            pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                        degree_thres_list, shift_thres_list)
            num_pred = pose_cls_pred_match.shape[2]
            pred_start = pose_pred_count[cls_id]
            pred_end = pred_start + num_pred
            pose_pred_count[cls_id] = pred_end
            pose_pred_matches_all[cls_id][:, :, pred_start:pred_end] = pose_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            assert cls_pred_scores_tile.shape[2] == num_pred
            pose_pred_scores_all[cls_id][:, :, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = pose_cls_gt_match.shape[2]
            gt_start = pose_gt_count[cls_id]
            gt_end = gt_start + num_gt
            pose_gt_count[cls_id] = gt_end
            pose_gt_matches_all[cls_id][:, :, gt_start:gt_end] = pose_cls_gt_match
            set_trace()
            
        set_trace()

    # trim zeros
    for cls_id in range(num_classes):
        # IoU
        iou_pred_matches_all[cls_id] = iou_pred_matches_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_pred_scores_all[cls_id] = iou_pred_scores_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_gt_matches_all[cls_id] = iou_gt_matches_all[cls_id][:, :iou_gt_count[cls_id]]
        # pose
        pose_pred_matches_all[cls_id] = pose_pred_matches_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_pred_scores_all[cls_id] = pose_pred_scores_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_gt_matches_all[cls_id] = pose_gt_matches_all[cls_id][:, :, :pose_gt_count[cls_id]]

    # compute 3D IoU mAP
    for cls_id in range(1, num_classes):
        for s, iou_thres in enumerate(iou_thres_list):
            iou_aps[cls_id, s], iou_acc[cls_id, s] = compute_ap_and_acc(iou_pred_matches_all[cls_id][s, :],
                                                                        iou_pred_scores_all[cls_id][s, :],
                                                                        iou_gt_matches_all[cls_id][s, :])
    iou_aps[-1, :] = np.mean(iou_aps[1:-1, :], axis=0)
    iou_acc[-1, :] = np.mean(iou_acc[1:-1, :], axis=0)
    # compute pose mAP
    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]
                pose_aps[cls_id, i, j], pose_acc[cls_id, i, j] = compute_ap_and_acc(cls_pose_pred_matches_all,
                                                                                    cls_pose_pred_scores_all,
                                                                                    cls_pose_gt_matches_all)
            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
            pose_acc[-1, i, j] = np.mean(pose_acc[1:-1, i, j])

    # save results to pkl
    result_dict = {}
    result_dict['iou_thres_list'] = iou_thres_list
    result_dict['degree_thres_list'] = degree_thres_list
    result_dict['shift_thres_list'] = shift_thres_list
    result_dict['iou_aps'] = iou_aps
    result_dict['pose_aps'] = pose_aps
    result_dict['iou_acc'] = iou_acc
    result_dict['pose_acc'] = pose_acc
    pkl_path = os.path.join(out_dir, 'mAP_Acc.pkl')
    with open(pkl_path, 'wb') as f:
        cPickle.dump(result_dict, f)
    return iou_aps, pose_aps, iou_acc, pose_acc
'''


def get_sRT(sRT, energy=None, error_mode='nearest', ratio=1.0):
    """ 
    Args:
        sRT (np.array): [ins_num, inference_num, 4, 4]
        error_mode (str): 'nearest', 'average'
        ratio (float): ratio of selection
    Returns:
        sRT (np.array): [ins_num, int(ratio*inference_num), 4, 4] or [ins_num, 4, 4]
    """
    ins_num = sRT.shape[0]
    repeat_num = int(sRT.shape[1] * ratio)
    repeat_num = max(1, repeat_num)
    selected_sRT = sRT[:, :repeat_num, :, :]
    selected_energy = energy[:, :repeat_num, :]
    
    if error_mode == 'nearest' or energy.shape[0] == 0: 
        reshaped_selected_sRT = selected_sRT.reshape(ins_num*repeat_num, 4, 4)
        quat_wxyz = pytorch3d.transforms.matrix_to_quaternion(torch.from_numpy(reshaped_selected_sRT[:, :3, :3])).cuda()
        quat_wxyz = torch.cat((quat_wxyz, torch.tensor(reshaped_selected_sRT[:, :3, 3]).to(quat_wxyz.device)), dim=-1)
        quat_wxyz = quat_wxyz.reshape(ins_num, repeat_num, -1)
        
        average_pred_pose = torch.zeros((quat_wxyz.shape[0], quat_wxyz.shape[-1])).to(quat_wxyz.device)
        average_pred_pose[:, :4] = average_quaternion_batch(quat_wxyz[:, :, :4])
        average_pred_pose[:, 4:] = torch.mean(quat_wxyz[:, :, 4:], dim=1)
        average_sRT = np.identity(4)[np.newaxis, ...].repeat(ins_num, 0)
        average_sRT[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(average_pred_pose[:, :4]).cpu().numpy()
        average_sRT[:, :3, 3] = average_pred_pose[:, 4:].cpu().numpy()
        return selected_sRT, average_sRT, selected_energy
    elif error_mode == 'average':
        return selected_sRT, selected_sRT[:, 0, :, :], selected_energy
    else:
        raise NotImplementedError
    

def get_average_sRT(sRT, energy=None, error_mode='nearest', ratio=1.0):
    """ 
    Args:
        sRT (np.array): [ins_num, inference_num, 4, 4]
        error_mode (str): 'nearest', 'average'
        ratio (float): ratio of selection
    Returns:
        sRT (np.array): [ins_num, int(ratio*inference_num), 4, 4] or [ins_num, 4, 4]
    """
    ins_num = sRT.shape[0]
    repeat_num = int(sRT.shape[1] * ratio)
    repeat_num = max(1, repeat_num)
    selected_sRT = sRT[:, :repeat_num, :, :]
    selected_energy = energy[:, :repeat_num, :]
    
    if error_mode == 'average':
        reshaped_selected_sRT = selected_sRT.reshape(ins_num*repeat_num, 4, 4)
        quat_wxyz = pytorch3d.transforms.matrix_to_quaternion(torch.from_numpy(reshaped_selected_sRT[:, :3, :3])).cuda()
        quat_wxyz = torch.cat((quat_wxyz, torch.tensor(reshaped_selected_sRT[:, :3, 3]).to(quat_wxyz.device)), dim=-1)
        quat_wxyz = quat_wxyz.reshape(ins_num, repeat_num, -1)
        
        average_pred_pose = torch.zeros((quat_wxyz.shape[0], quat_wxyz.shape[-1])).to(quat_wxyz.device)
        average_pred_pose[:, :4] = average_quaternion_batch(quat_wxyz[:, :, :4])
        average_pred_pose[:, 4:] = torch.mean(quat_wxyz[:, :, 4:], dim=1)
        average_sRT = np.identity(4)[np.newaxis, ...].repeat(ins_num, 0)
        average_sRT[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(average_pred_pose[:, :4]).cpu().numpy()
        average_sRT[:, :3, 3] = average_pred_pose[:, 4:].cpu().numpy()
        return selected_sRT, average_sRT, selected_energy
    elif error_mode == 'nearest': 
        return selected_sRT, selected_sRT[:, 0, :, :], selected_energy
    else:
        raise NotImplementedError


def sort_sRT_by_energy(sRT, energy=None, RT_overlaps=None, ranker='energy_ranker', ratio=1.0, error_mode='average'):
    """ Sort multi hypothesis sRT by a ranker
    
    Args:
        sRT (np.array): [pred_num, repeat_num, 4, 4]
        energy (np.array): [pred_num, repeat_num, 2]
        RT_overlaps (np.array): [pred_num, gt_num, repeat_num, 2] the last axis is rotation error and translation respectively.
        ranker (str): energy_raker, gt_ranker or random
        ratio (float)
        
    Return:
        sorted_sRT (np.array): [pred_num, repeat_num, 4, 4]
    """
    pred_num = sRT.shape[0]
    repeat_num = sRT.shape[1]
    selected_num = max(1, int(repeat_num * ratio))
    
    if sRT.shape[0] == 0:
        return sRT[:, :selected_num, :, :], None, None
    
    ''' get energy '''
    if ranker == 'energy_ranker':
        RT_energy = energy
    elif ranker == 'gt_ranker':
        # set_trace()
        RT_energy = -np.min(RT_overlaps, axis=1)
        # R_energy = np.min(RT_overlaps[..., 0], axis=0)[..., np.newaxis]
        # T_energy = np.min(RT_overlaps[..., 1], axis=1)[..., np.newaxis]
        # RT_energy = np.concatenate((R_energy, T_energy), axis=-1)
    elif ranker == 'random':
        RT_energy = np.random.rand(pred_num, repeat_num, 2)
    else:
        raise NotImplementedError
    
    ''' sort sRT and energy '''
    sorted_sRT, sorted_RT_energy = sort_sRT(sRT, RT_energy)
    selected_sRT = sorted_sRT[:, :selected_num, :, :]
    selected_energy = sorted_RT_energy[:, :selected_num, :]
    
    ''' calculate averaged sRT of selected sRT '''
    if error_mode == 'average':
        reshaped_selected_sRT = selected_sRT.reshape(pred_num*selected_num, 4, 4)
        reshaped_selected_sRT = torch.from_numpy(reshaped_selected_sRT).cuda()
        quat_wxyz = pytorch3d.transforms.matrix_to_quaternion(reshaped_selected_sRT[:, :3, :3])
        quat_wxyz = torch.cat((quat_wxyz, reshaped_selected_sRT[:, :3, 3]), dim=-1)
        quat_wxyz = quat_wxyz.reshape(pred_num, selected_num, -1)
        
        average_pred_pose = torch.zeros((quat_wxyz.shape[0], quat_wxyz.shape[-1])).to(quat_wxyz.device)
        average_pred_pose[:, :4] = average_quaternion_batch(quat_wxyz[:, :, :4])
        average_pred_pose[:, 4:] = torch.mean(quat_wxyz[:, :, 4:], dim=1)
        average_sRT = np.identity(4)[np.newaxis, ...].repeat(pred_num, 0)
        average_sRT[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(average_pred_pose[:, :4]).cpu().numpy()
        average_sRT[:, :3, 3] = average_pred_pose[:, 4:].cpu().numpy()
        return selected_sRT, average_sRT, selected_energy
    elif error_mode == 'nearest':
        return selected_sRT, None, selected_energy
    else:
        raise NotImplementedError


def compute_mAP(pred_results, out_dir, degree_thresholds=[180], shift_thresholds=[100],
                iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False,
                repeat_num=None, pooling_mode='average', ratio=1.0, so3_vis=False, ranker='energy_ranker'):
    """ Compute mean Average Precision.

    Returns:
        iou_aps:
        pose_aps:
        iou_acc:
        pose_acc:
    """
    
    synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)
    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    # pre-allocate more than enough memory
    iou_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_acc = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 30000)) for _ in range(num_classes)]
    iou_pred_count = [0 for _ in range(num_classes)]
    iou_gt_count = [0 for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_acc = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_gt_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 30000)) for _ in range(num_classes)]
    pose_pred_count = [0 for _ in range(num_classes)]
    pose_gt_count = [0 for _ in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(tqdm(pred_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_sRT = np.array(result['gt_RTs'])
        gt_size = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']
        gt_bboxes = np.array(result['gt_bboxes'])

        pred_class_ids = result['pred_class_ids']
        multi_hypothesis_pred_sRT = result['multi_hypothesis_pred_RTs']
        pred_energy = result['energy']
        # multi_hypothesis_pred_sRT, pred_sRT, pred_energy = get_sRT(
        #     sRT=result['multi_hypothesis_pred_RTs'], 
        #     energy=result['energy'],
        #     error_mode=error_mode, 
        #     ratio=ratio)
        pred_size = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_bboxes = np.array(result['pred_bboxes'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue


        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_sRT = gt_sRT[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))
            # cls_gt_size = gt_size[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            cls_gt_bboxes = gt_bboxes[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4))
            
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            
            cls_multi_hypothesis_pred_sRT = multi_hypothesis_pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, repeat_num, 4, 4))
            cls_pred_energy = pred_energy[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, repeat_num, 2))
            # cls_pred_sRT = pred_sRT[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            
            # cls_pred_size = pred_size[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4))
            
            # calculate the overlap between each gt instance and pred instance
            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = \
                compute_2d_IoU_matches(cls_gt_class_ids, cls_gt_bboxes,
                                       cls_pred_class_ids, cls_pred_bboxes, cls_pred_scores,
                                       iou_thres_list)

            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_multi_hypothesis_pred_sRT = cls_multi_hypothesis_pred_sRT[iou_pred_indices]
                cls_pred_energy = cls_pred_energy[iou_pred_indices]
                # cls_pred_sRT = cls_pred_sRT[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
            
            num_pred = iou_cls_pred_match.shape[1]
            pred_start = iou_pred_count[cls_id]
            pred_end = pred_start + num_pred
            iou_pred_count[cls_id] = pred_end
            iou_pred_matches_all[cls_id][:, pred_start:pred_end] = iou_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            assert cls_pred_scores_tile.shape[1] == num_pred
            iou_pred_scores_all[cls_id][:, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = iou_cls_gt_match.shape[1]
            gt_start = iou_gt_count[cls_id]
            gt_end = gt_start + num_gt
            iou_gt_count[cls_id] = gt_end
            iou_gt_matches_all[cls_id][:, gt_start:gt_end] = iou_cls_gt_match
            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)
                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                
                cls_multi_hypothesis_pred_sRT = cls_multi_hypothesis_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, repeat_num, 4, 4))
                cls_pred_energy = cls_pred_energy[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, repeat_num, 2))
                # cls_pred_sRT = cls_pred_sRT[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                        
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_sRT = cls_gt_sRT[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)

            
            # RT_overlaps_list = []
            # inference_num = cls_multi_hypothesis_pred_sRT.shape[1]
            # for i in range(10):
            #     RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
            #                 cls_pred_class_ids, cls_multi_hypothesis_pred_sRT[:, i, ...], synset_names)
            #     RT_overlaps_list.append(RT_overlaps)
            # RT_overlaps_test = np.min(np.array(RT_overlaps_list), axis=0)
            
            cls_multi_hypothesis_RT_overlaps = None
            if ranker == 'gt_ranker':
                RT_overlaps_list = []
                inference_num = cls_multi_hypothesis_pred_sRT.shape[1]
                for i in range(inference_num):
                    RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                cls_pred_class_ids, cls_multi_hypothesis_pred_sRT[:, i, ...], synset_names)
                    RT_overlaps_list.append(RT_overlaps)
                cls_multi_hypothesis_RT_overlaps = np.array(RT_overlaps_list).transpose(1, 2, 0, 3)
                # RT_overlaps = np.min(np.array(RT_overlaps_list), axis=0)

            if pooling_mode == 'nearest': 
                ''' sort results by ranker '''
                sorted_cls_multi_hypothesis_pred_sRT, _, sorted_cls_pred_energy = \
                    sort_sRT_by_energy(cls_multi_hypothesis_pred_sRT, cls_pred_energy, cls_multi_hypothesis_RT_overlaps, ranker, ratio, 'nearest')
                RT_overlaps_list = []
                inference_num = sorted_cls_multi_hypothesis_pred_sRT.shape[1]
                for i in range(inference_num):
                    RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                cls_pred_class_ids, sorted_cls_multi_hypothesis_pred_sRT[:, i, ...], synset_names)
                    RT_overlaps_list.append(RT_overlaps)
                RT_overlaps = np.min(np.array(RT_overlaps_list), axis=0)

            elif pooling_mode == 'average':       
                ''' sort results by ranker '''
                sorted_cls_multi_hypothesis_pred_sRT, average_cls_pred_sRT, sorted_cls_pred_energy = \
                    sort_sRT_by_energy(cls_multi_hypothesis_pred_sRT, cls_pred_energy, cls_multi_hypothesis_RT_overlaps, ranker, ratio, 'average')
            
                ''' calculate rotation and translation error'''
                RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_sRT, cls_gt_handle_visibility,
                                cls_pred_class_ids, average_cls_pred_sRT, synset_names)
            else:
                raise NotImplementedError
            
            if so3_vis and cls_gt_sRT.shape[0] > 0:
                so3_visualization(
                    pred_rot=sorted_cls_multi_hypothesis_pred_sRT[0, :, :3, :3],
                    # energy=sorted_cls_pred_energy[0],
                    gt_rot=cls_gt_sRT[0, :3, :3]/np.cbrt(np.linalg.det(cls_gt_sRT[0, :3, :3])))
                set_trace()
            
            pose_cls_gt_match, pose_cls_pred_match = compute_RT_matches(RT_overlaps, cls_pred_class_ids, cls_gt_class_ids,
                                                                        degree_thres_list, shift_thres_list)
            num_pred = pose_cls_pred_match.shape[2]
            pred_start = pose_pred_count[cls_id]
            pred_end = pred_start + num_pred
            pose_pred_count[cls_id] = pred_end
            pose_pred_matches_all[cls_id][:, :, pred_start:pred_end] = pose_cls_pred_match
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            assert cls_pred_scores_tile.shape[2] == num_pred
            pose_pred_scores_all[cls_id][:, :, pred_start:pred_end] = cls_pred_scores_tile
            num_gt = pose_cls_gt_match.shape[2]
            gt_start = pose_gt_count[cls_id]
            gt_end = gt_start + num_gt
            pose_gt_count[cls_id] = gt_end
            pose_gt_matches_all[cls_id][:, :, gt_start:gt_end] = pose_cls_gt_match


    # trim zeros
    for cls_id in range(num_classes):
        # IoU
        iou_pred_matches_all[cls_id] = iou_pred_matches_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_pred_scores_all[cls_id] = iou_pred_scores_all[cls_id][:, :iou_pred_count[cls_id]]
        iou_gt_matches_all[cls_id] = iou_gt_matches_all[cls_id][:, :iou_gt_count[cls_id]]
        # pose
        pose_pred_matches_all[cls_id] = pose_pred_matches_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_pred_scores_all[cls_id] = pose_pred_scores_all[cls_id][:, :, :pose_pred_count[cls_id]]
        pose_gt_matches_all[cls_id] = pose_gt_matches_all[cls_id][:, :, :pose_gt_count[cls_id]]

    # compute 3D IoU mAP
    for cls_id in range(1, num_classes):
        for s, iou_thres in enumerate(iou_thres_list):
            iou_aps[cls_id, s], iou_acc[cls_id, s] = compute_ap_and_acc(iou_pred_matches_all[cls_id][s, :],
                                                                        iou_pred_scores_all[cls_id][s, :],
                                                                        iou_gt_matches_all[cls_id][s, :])
    iou_aps[-1, :] = np.mean(iou_aps[1:-1, :], axis=0)
    iou_acc[-1, :] = np.mean(iou_acc[1:-1, :], axis=0)
    # compute pose mAP
    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]
                pose_aps[cls_id, i, j], pose_acc[cls_id, i, j] = compute_ap_and_acc(cls_pose_pred_matches_all,
                                                                                    cls_pose_pred_scores_all,
                                                                                    cls_pose_gt_matches_all)
            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
            pose_acc[-1, i, j] = np.mean(pose_acc[1:-1, i, j])

    # save results to pkl
    result_dict = {}
    result_dict['iou_thres_list'] = iou_thres_list
    result_dict['degree_thres_list'] = degree_thres_list
    result_dict['shift_thres_list'] = shift_thres_list
    result_dict['iou_aps'] = iou_aps
    result_dict['pose_aps'] = pose_aps
    result_dict['iou_acc'] = iou_acc
    result_dict['pose_acc'] = pose_acc
    pkl_path = os.path.join(out_dir, 'mAP_Acc.pkl')
    with open(pkl_path, 'wb') as f:
        cPickle.dump(result_dict, f)
    return iou_aps, pose_aps, iou_acc, pose_acc


def plot_mAP(iou_aps, pose_aps, out_dir, iou_thres_list, degree_thres_list, shift_thres_list, out_name='mAP.png'):
    """ Draw iou 3d AP vs. iou thresholds.
    """

    labels = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'mean', 'nocs']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:purple', 'tab:red', 'tab:gray']
    styles = ['-', '-', '-', '-', '-', '-', '--', ':']

    fig, (ax_iou, ax_degree, ax_shift) = plt.subplots(1, 3, figsize=(8, 3.5))
    # IoU subplot
    ax_iou.set_title('3D IoU', fontsize=10)
    ax_iou.set_ylabel('Average Precision')
    ax_iou.set_ylim(0, 100)
    ax_iou.set_xlabel('Percent')
    ax_iou.set_xlim(0, 100)
    ax_iou.xaxis.set_ticks([0, 25, 50, 75, 100])
    ax_iou.grid()
    for i in range(1, iou_aps.shape[0]):
        ax_iou.plot(100*np.array(iou_thres_list), 100*iou_aps[i, :],
                    color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # rotation subplot
    ax_degree.set_title('Rotation', fontsize=10)
    ax_degree.set_ylim(0, 100)
    ax_degree.yaxis.set_ticklabels([])
    ax_degree.set_xlabel('Degree')
    # ax_degree.set_xlim(0, 45)
    # ax_degree.xaxis.set_ticks([0, 15, 30, 45])
    ax_degree.set_xlim(0, 10)
    ax_degree.xaxis.set_ticks([0, 5, 10])
    ax_degree.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_degree.plot(np.array(degree_thres_list), 100*pose_aps[i, :len(degree_thres_list), -1],
                       color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    # translation subplot
    ax_shift.set_title('Translation', fontsize=10)
    ax_shift.set_ylim(0, 100)
    ax_shift.yaxis.set_ticklabels([])
    ax_shift.set_xlabel('Centimeter')
    # ax_shift.set_xlim(0, 10)
    # ax_shift.xaxis.set_ticks([0, 5, 10])
    ax_shift.set_xlim(0, 6)
    ax_shift.xaxis.set_ticks([0, 2, 4, 6])
    ax_shift.grid()
    for i in range(1, pose_aps.shape[0]):
        ax_shift.plot(np.array(shift_thres_list), 100*pose_aps[i, -1, :len(shift_thres_list)],
                      color=colors[i-1], linestyle=styles[i-1], label=labels[i-1])
    ax_shift.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, out_name),dpi=600)
    plt.close(fig)
    return


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT


def draw_coord_axis(img, img_pts):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    print(img_pts.shape)
    color_x_axis = (255, 0, 0)
    color_y_axis = (0, 255, 0)
    color_z_axis = (0, 0, 255)
    
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), color_x_axis, 2)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), color_y_axis, 2)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), color_z_axis, 2)
    
    return img  


def draw_coord_axis_with_error(img, img_pts, flag):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    color_x_axis = (255, 0, 0)
    color_y_axis = (0, 255, 0)
    color_z_axis = (0, 0, 255)
    
    if flag:
        img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), color_x_axis, 2)
        img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), color_y_axis, 2)
        img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), color_z_axis, 2)
    else:
        img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), color_z_axis, 2)
        img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), color_z_axis, 2)
        img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), color_z_axis, 2)        
    
    return img  


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img


def draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids, nocs_sRT, nocs_size, nocs_class_ids, draw_gt=True, draw_nocs=True, draw_pred=True, draw_axis=True, draw_bbox=False):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    # draw nocs results - BLUE color
    if draw_nocs:
        for i in range(nocs_sRT.shape[0]):
            if nocs_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(nocs_sRT[i, :, :])
            else:
                sRT = nocs_sRT[i, :, :]
            bbox_3d = get_3d_bbox(nocs_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0))
            
    # darw ground truth - GREEN color
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            if draw_bbox:
                bbox_3d = get_3d_bbox(gt_size[i, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                img = draw_bboxes(img, projected_bbox, (0, 255, 0))
            if draw_axis:
                coord_axis_3d = get_3d_coord_axis()
                RT = copy.deepcopy(sRT)
                RT[:3, :3] = RT[:3, :3] / np.linalg.norm(RT[:3, 0])
                transformed_coord_axis_3d = transform_coordinates_3d(coord_axis_3d, RT)
                projected_coord_axis_3d = calculate_2d_projections(transformed_coord_axis_3d, intrinsics)
                img = draw_coord_axis(img, projected_coord_axis_3d)                
            
    # darw prediction - RED color
    if draw_pred:
        for i in range(pred_sRT.shape[0]):
            if pred_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(pred_sRT[i, :, :])
            else:
                sRT = pred_sRT[i, :, :]
            if draw_bbox:
                bbox_3d = get_3d_bbox(pred_size[i, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                img = draw_bboxes(img, projected_bbox, (0, 0, 255))
            if draw_axis:
                coord_axis_3d = get_3d_coord_axis()
                RT = copy.deepcopy(sRT)
                RT[:3, :3] = RT[:3, :3] / np.linalg.norm(RT[:3, 0])
                transformed_coord_axis_3d = transform_coordinates_3d(coord_axis_3d, RT)
                projected_coord_axis_3d = calculate_2d_projections(transformed_coord_axis_3d, intrinsics)
                img = draw_coord_axis(img, projected_coord_axis_3d) 
    #print(img.shape)            
    cv2.imwrite(out_path, img)
    # cv2.imshow('vis', img)
    # cv2.waitKey(0)

    
def draw_detections_with_error(img, flag_list, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids, nocs_sRT, nocs_size, nocs_class_ids, draw_gt=True, draw_nocs=True, draw_pred=True, draw_axis=True, draw_bbox=False):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    # draw nocs results - BLUE color
    if draw_nocs:
        for i in range(nocs_sRT.shape[0]):
            if nocs_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(nocs_sRT[i, :, :])
            else:
                sRT = nocs_sRT[i, :, :]
            bbox_3d = get_3d_bbox(nocs_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0))
            
    # darw ground truth - GREEN color
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            if draw_bbox:
                bbox_3d = get_3d_bbox(gt_size[i, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                img = draw_bboxes(img, projected_bbox, (0, 255, 0))
            if draw_axis:
                coord_axis_3d = get_3d_coord_axis()
                RT = copy.deepcopy(sRT)
                RT[:3, :3] = RT[:3, :3] / np.linalg.norm(RT[:3, 0])
                transformed_coord_axis_3d = transform_coordinates_3d(coord_axis_3d, RT)
                projected_coord_axis_3d = calculate_2d_projections(transformed_coord_axis_3d, intrinsics)
                img = draw_coord_axis(img, projected_coord_axis_3d)                
            
    # darw prediction - RED color
    if draw_pred:
        for i in range(pred_sRT.shape[0]):
            if pred_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(pred_sRT[i, :, :])
            else:
                sRT = pred_sRT[i, :, :]
            if draw_bbox:
                bbox_3d = get_3d_bbox(pred_size[i, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                img = draw_bboxes(img, projected_bbox, (0, 0, 255))
            if draw_axis:
                coord_axis_3d = get_3d_coord_axis()
                RT = copy.deepcopy(sRT)
                RT[:3, :3] = RT[:3, :3] / np.linalg.norm(RT[:3, 0])
                transformed_coord_axis_3d = transform_coordinates_3d(coord_axis_3d, RT)
                projected_coord_axis_3d = calculate_2d_projections(transformed_coord_axis_3d, intrinsics)
                img = draw_coord_axis_with_error(img, projected_coord_axis_3d, flag_list[i]) 
    #print(img.shape)            
    cv2.imwrite(out_path, img)


def compute_sRT_flags(pred_RTs, gt_RTs, gt_class_ids, gt_handle_visibility, synset_names, R_thresh=10, T_thresh=10):
    flag_list = []
    num_preds, _, _ = pred_RTs.shape
    num_gts, _, _ = gt_RTs.shape
    for m in range(num_preds):
        pred_RT = pred_RTs[m]
        flag = False
        for n in range(num_gts):
             gt_RT = gt_RTs[n]
             #R_error, T_error, _ = compute_RT_errors(pred_RT, gt_RT) compute_RT_errors(sRT_1, sRT_2, class_id, handle_visibility, synset_names)
             #print("gt_class_ids[n]", gt_class_ids[n])
             R_error, T_error = compute_RT_errors(pred_RT, gt_RT, gt_class_ids[n]-1,gt_handle_visibility[n], synset_names)
             if R_error < R_thresh and T_error < T_thresh:
                 flag = True
                 break
        
        flag_list.append(flag)
    return flag_list
             


if __name__ == '__main__':
    bs = 2
    inference_num = 4
    poses = np.random.rand(bs, inference_num, 4, 4)
    energy = np.random.rand(bs, inference_num, 2)
    sorted_poses, _, sorted_energy = sort_sRT_by_energy(poses, energy, error_mode='nearest', ratio=0.5)


    
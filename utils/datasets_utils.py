import numpy as np
import cv2

def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(0, width-1, width, dtype=np.float32)
    y = np.linspace(0, height-1, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy


def aug_bbox_DZI(hyper_params, bbox_xyxy, im_H, im_W):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    if hyper_params['DZI_TYPE'].lower() == "uniform":
        scale_ratio = 1 + hyper_params['DZI_SCALE_RATIO'] * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = hyper_params['DZI_SHIFT_RATIO'] * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * hyper_params['DZI_PAD_SCALE']
    elif hyper_params['DZI_TYPE'].lower() == "roi10d":
        # shift (x1,y1), (x2,y2) by 15% in each direction
        _a = -0.15
        _b = 0.15
        x1 += bw * (np.random.rand() * (_b - _a) + _a)
        x2 += bw * (np.random.rand() * (_b - _a) + _a)
        y1 += bh * (np.random.rand() * (_b - _a) + _a)
        y2 += bh * (np.random.rand() * (_b - _a) + _a)
        x1 = min(max(x1, 0), im_W)
        x2 = min(max(x1, 0), im_W)
        y1 = min(max(y1, 0), im_H)
        y2 = min(max(y2, 0), im_H)
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        scale = max(y2 - y1, x2 - x1) * hyper_params['DZI_PAD_SCALE']
    elif hyper_params['DZI_TYPE'].lower() == "truncnorm":
        raise NotImplementedError("DZI truncnorm not implemented yet.")
    else:
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale


def aug_bbox_eval(bbox_xyxy, im_H, im_W):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
        center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    bbox_center = np.array([cx, cy])  # (w/2, h/2)
    scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

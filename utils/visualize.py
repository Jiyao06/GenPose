import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from ipdb import set_trace
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image, make_grid

os.sys.path.append('..')

from utils.misc import exists_or_mkdir, get_rot_matrix, transform_batch_pts
from utils.so3_visualize import visualize_so3


def resize_img_keep_ratio(img, target_size):
    old_size= img.shape[0:2]
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv2.resize(img,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1] 
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0)) 
    return img_new


def get_mat_from_A_to_B(pts_A, pts_B):
    """
    Function: get transformation matrix form point clouds A to point clouds B
    Args:
        pts_A: source points
        pts_B: target points
    Returns:
        R: rotation matrix from pts_A to pts_B
        T: translation matrix from pts_A to pts_B
    """
    muA = np.mean(pts_A, axis=0)
    muB = np.mean(pts_B, axis=0)

    zero_mean_A = pts_A - muA
    zero_mean_B = pts_B - muB

    # calculate the covariance matrix
    covMat = np.matmul(np.transpose(zero_mean_A), zero_mean_B)
    U, S, Vt = np.linalg.svd(covMat)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    T = (-np.matmul(R, muA.T) + muB.T).reshape(3, 1)
    return R, T


def get_camera_pose(start_point, look_at, up):
    """
    Function: get camera pose form the particular representation
    Args:
        start_point: location of camera, [3]
        look_at: the look_at point of camera, [3]
        up: the y axis of camera, [3]
    Returns:
        R: rotation matrix of camera
        T: translation matrix of camera
    """
    up = up / np.linalg.norm(up)
        
    vector_z = (look_at - start_point)
    vector_z = vector_z/np.linalg.norm(vector_z)
    
    vector_x = np.cross(up, vector_z)
    vector_x = vector_x/np.linalg.norm(vector_x)
    
    vector_y = np.cross(vector_z, vector_x)
    vector_y = vector_y/np.linalg.norm(vector_y)

    # points in camera coordinates
    point_sensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

    # points in world coordinates 
    point_world = np.array([start_point,
                           start_point + vector_x,
                           start_point + vector_y * 2,
                           start_point + vector_z * 3])

    R, T = get_mat_from_A_to_B(point_sensor, point_world)
    return R, T


def proj_uv_to_image(proj_uv, color=None, image_size={'xres':640, 'yres':360}):
    """
    Function: convert projected uv to image
    Args:
        proj_uv: project_uv, [N, 2]
        color: None or [N, 3]
        image_size: the size of output image
    Returns:
        image: [xres, yres, 3]
    """
    uv = np.around(proj_uv / proj_uv[:, 2][:, np.newaxis]).astype(int)[:, :2]
    uv[:, 0][uv[:, 0] < 0] = 0
    uv[:, 0][uv[:, 0] > image_size['xres'] - 1] = image_size['xres'] - 1
    uv[:, 1][uv[:, 1] < 0] = 0
    uv[:, 1][uv[:, 1] > image_size['yres'] - 1] = image_size['yres'] - 1
    
    image = np.ones([image_size['yres'], image_size['xres'], 3]).astype(np.uint8)
    image *= 255
    if color is None:
        image[uv[:, 1], uv[:, 0]] = np.array([0, 0, 255])
    else:
        image[uv[:, 1], uv[:, 0]] = color
    return image


def project_pts_to_image(pts, 
                         image_size={'xres':640, 'yres':360},
                         camera_intrinsics={'fx': 502.30, 'fy': 502.30, 'cx': 319.5, 'cy': 179.5, 'xres': 640, 'yres': 360}, 
                         camera_extrinsics={'look_at': np.array([0, 0, 1]), 'location': np.array([0, 0, 0]), 'up': np.array([0, 1, 0])}):
    """
    Function: render points
    Args:
        pts: input points. wtih color: [N, 6], w/o color: [N, 3]
        image_size: the size of output image
        camera_intrinsics: dict
        camera_extrinsics: dict
    Returns:
        image: [xres, yres, 3]
    """
    # pts, N*_ (N*3/6)

    x_scale = image_size['xres'] / camera_intrinsics['xres']
    y_scale = image_size['yres'] / camera_intrinsics['yres']
    
    fx = camera_intrinsics['fx'] * x_scale
    fy = camera_intrinsics['fy'] * y_scale
    cx = camera_intrinsics['cx'] * x_scale
    cy = camera_intrinsics['cy'] * y_scale
    
    cam_intrinsic_mat = np.array([[fx,  0, cx],
                                  [ 0, fy, cy],
                                  [ 0,  0,  1]])
    cam_R, cam_T = get_camera_pose(camera_extrinsics['location'],
                                   camera_extrinsics['look_at'],
                                   camera_extrinsics['up'])
    trans_c2w = np.concatenate((cam_R, cam_T), axis=1)
    trans_c2w = np.concatenate((trans_c2w, np.array([[0, 0, 0, 1]])), axis=0)
    trans_w2c = np.linalg.inv(trans_c2w)
    
    if pts.shape[1] == 6:
        color = pts[:, 3:]
    else:
        color = None
        
    pts_in_cam_space = trans_w2c @ np.concatenate((pts[:, :3], np.ones([pts.shape[0], 1])), axis=1).T
    uv = cam_intrinsic_mat @ pts_in_cam_space[:3, :]
    uv = uv.T
    image = proj_uv_to_image(uv, color, image_size)
        
    return image


def pts_visulize(pts):
    image_size={'xres':360, #640, 
                'yres':360
                }
    
    cam_intrinsics={'fx': 502.30, 
                    'fy': 502.30, 
                    'cx': 319.5, 
                    'cy': 319.5, #179.5, 
                    'xres': 640,  
                    'yres': 640  # 360
                    }
    
    cam_top_view={'look_at': np.array([0, 0, 0]), 
                  'location': np.array([0, 0.5, 0]), 
                  'up': np.array([0, 0, -1])}
    
    cam_front_view={'look_at': np.array([0, 0, 0]), 
                    'location': np.array([0, 0, -0.5]), 
                    'up': np.array([0, -1, 0])}   

    top_view_image = project_pts_to_image(pts=pts,
                                          image_size=image_size,
                                          camera_intrinsics=cam_intrinsics,
                                          camera_extrinsics=cam_top_view)       

    front_view_image = project_pts_to_image(pts=pts,
                                            image_size=image_size,
                                            camera_intrinsics=cam_intrinsics,
                                            camera_extrinsics=cam_front_view)      
    return front_view_image, top_view_image

    
def create_grid_image(batch_pts, batch_pred_pose, batch_gt_pose, batch_color, pose_mode='quat_wxyz', inverse_pose=False):
    B = batch_pts.shape[0]
    max_image_num = min(B, 16)
    if B > max_image_num:
        batch_pts = batch_pts[:16]
        batch_pred_pose = batch_pred_pose[:16]
        if not batch_gt_pose is None:
            batch_gt_pose = batch_gt_pose[:16]
        if not batch_color is None:
            batch_color = batch_color[:16]        
            batch_color_array = batch_color.cpu().numpy()
    
    color_image_list = []
    pred_pts = transform_batch_pts(batch_pts, batch_pred_pose, pose_mode, inverse_pose).cpu().numpy()
    pred_front_view_image_list = []
    pred_top_view_image_list = []
    
    if not batch_gt_pose is None:
        gt_pts = transform_batch_pts(batch_pts, batch_gt_pose, pose_mode, inverse_pose).cpu().numpy()
        gt_front_view_image_list = []
        gt_top_view_image_list = []
    
    for i in range(max_image_num):
        pred_front_view_image, pred_top_view_image = pts_visulize(pred_pts[i])
        pred_front_view_image_list.append(pred_front_view_image)
        pred_top_view_image_list.append(pred_top_view_image)
        
        if not batch_gt_pose is None:
            gt_front_view_image, gt_top_view_image = pts_visulize(gt_pts[i])
            gt_front_view_image_list.append(gt_front_view_image)
            gt_top_view_image_list.append(gt_top_view_image)  

        if not batch_color is None:
            size = [pred_top_view_image.shape[0], pred_top_view_image.shape[1]]
            resized_color = resize_img_keep_ratio(batch_color_array[i].transpose((1, 2, 0)), size)
            color_image_list.append(resized_color)
              
    pred_front_view_tensor = torch.from_numpy(np.array(pred_front_view_image_list)).permute(0, 3, 1, 2)
    pred_top_view_tensor = torch.from_numpy(np.array(pred_top_view_image_list)).permute(0, 3, 1, 2)
    
    if not batch_gt_pose is None:
        gt_front_view_tensor = torch.from_numpy(np.array(gt_front_view_image_list)).permute(0, 3, 1, 2)
        gt_top_view_tensor = torch.from_numpy(np.array(gt_top_view_image_list)).permute(0, 3, 1, 2)
    if not batch_color is None:
        color_tensor = torch.from_numpy(np.array(color_image_list)).permute(0, 3, 1, 2)
        images = torch.cat((color_tensor, pred_front_view_tensor, 
                            pred_top_view_tensor, 
                            gt_front_view_tensor, 
                            gt_top_view_tensor), dim=3)
    elif not batch_gt_pose is None:
        images = torch.cat((pred_front_view_tensor, 
                            pred_top_view_tensor, 
                            gt_front_view_tensor, 
                            gt_top_view_tensor), dim=3)
    else:
        images = torch.cat((pred_front_view_tensor, 
                            pred_top_view_tensor), dim=3)

    # grid_image = make_grid(images, 1, normalize=True, scale_each=True)
    grid_image = make_grid(images, 1, normalize=False, scale_each=False)

    return grid_image, images


def save_video(save_path, batch_pts, batch_pred_pose_list, batch_gt_pose, batch_color, fps, pose_mode='quat', inverse_pose=False):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    col_num = 5
    col_num = col_num - 1 if batch_color is None else col_num
    col_num = col_num - 2 if batch_gt_pose is None else col_num
    image_size = (360*col_num, 360)

    out_list = []
    for i in range(batch_pts.shape[0]):
        video_save_path = os.path.join(save_path, f'example_{str(i)}.mp4')
        out = cv2.VideoWriter(video_save_path, fourcc, fps, image_size, True)
        out_list.append(out)
    for _, batch_pose in enumerate(batch_pred_pose_list):
        _, images = create_grid_image(batch_pts, 
                                    batch_pose, 
                                    batch_gt_pose, 
                                    batch_color, 
                                    pose_mode,
                                    inverse_pose)

        
        for i in range(images.shape[0]):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            if image_size[0] != image.shape[1] or image_size[1] != image.shape[0]:
                raise Exception("Image size doesn't match!")
            out_list[i].write(image)
            
    for out in out_list:
        out.release()


def test_time_visulize(save_path, data, res, in_process_sample, pose_mode, o2c_pose):
    in_process_sample = in_process_sample.permute(1, 0, 2)
    exists_or_mkdir(save_path)
    # save res-grid
    pts = data['pts'] if 'pts_color' not in data.keys() else torch.cat((data['pts'], data['pts_color']), dim=2)
    bs = pts.shape[0]
    max_save_num = 16 if bs > 16 else pts.shape[0]
    grid_image, _ = create_grid_image(
        batch_pts=pts[:max_save_num] if bs > 1 else pts,
        batch_pred_pose=res[:max_save_num] if bs > 1 else res,
        batch_gt_pose=None if 'gt_pose' not in data.keys() else data['gt_pose'][:max_save_num] if bs > 1 else data['gt_pose'],
        batch_color=None,
        pose_mode=pose_mode,
        inverse_pose=o2c_pose
    )
    save_image(grid_image/255., os.path.join(save_path, 'res_grid.png'))
    # save in-process videos
    in_process_sample = in_process_sample[-in_process_sample.shape[0]//2:]  # only vis the last 50% states
    if in_process_sample.shape[0] > 100:
        interval = in_process_sample.shape[0] // 100    # we only render 100 images for each video
    else:
        interval = 1
    in_process_sample = in_process_sample[::interval, :, :]     # [100, num_videos, pose_dim]
    save_video(
        save_path=save_path,
        batch_pts=pts[:max_save_num] if bs > 1 else pts,
        batch_pred_pose_list=in_process_sample[:, :max_save_num] if bs > 1 else in_process_sample,
        batch_gt_pose=None if 'gt_pose' not in data.keys() else data['gt_pose'][:max_save_num] if bs > 1 else data['gt_pose'],
        batch_color=None,
        fps=in_process_sample.shape[0]//5,
        pose_mode=pose_mode,
        inverse_pose=o2c_pose
    ) # 5 sec for each video


def show_point_cloud(points, axis_size = 10, window_name = 'Open3D', colors = None):
    import open3d as o3d
    
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    if isinstance(points, list):
        pcds = []
        for i in range(len(points)):
            point_cloud = points[i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            if colors is not None:
                color = np.tile(colors[i], point_cloud.shape[0]).reshape(-1,3)
                pcd.colors = o3d.utility.Vector3dVector(color)
            pcds.append(pcd)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            color = np.tile(colors, points.shape[0]).reshape(-1,3)
            pcd.colors = o3d.utility.Vector3dVector(color)
        pcds = [pcd]
    o3d.visualization.draw_geometries(pcds + [axis_pcd], window_name = window_name)


def so3_visualization(pred_rot, energy=None, gt_rot=None):

    ''' ToDo: render pointcloud '''
    # grid_iamge, _ = create_grid_image(
    #     results['pts'][i].unsqueeze(0), 
    #     results['average_pred_pose'][i].unsqueeze(0), 
    #     results['gt_pose'][i].unsqueeze(0), 
    #     None, 
    #     pose_mode='quat_wxyz', 
    #     inverse_pose=cfg.o2c_pose,
    # )
    ''' so3 distribution visualization '''
    if energy is None:
        confidence = np.ones(pred_rot.shape[0]) / 200
    else:
        confidence = energy[:, 0] - np.mean(energy[:, 0])
        confidence = torch.softmax(torch.from_numpy(confidence), dim=0).cpu().numpy() / 10
    visualize_so3(
        save_path='./so3_distribution.png', 
        pred_rotations=pred_rot,
        # pred_rotation=pred_rot[0],
        gt_rotation=gt_rot,
        image=None,
        probabilities=confidence
        )

    

def generate_xml_for_mitsuba(
    pts, 
    image_size={'xres':640, 'yres':360},
    camera_intrinsic={'fov': 20},
    camera_extrinsic={'location': "3,3,3", 'look_at': "0,0,0", 'up': "0,0,1"},
    light_extrinsic={'location': "-4,4,20", 'look_at': "0,0,0", 'up': "0,0,1"}):
    
    def standardize_bbox(pcl, points_per_object):
        if pcl.shape[0] > points_per_object: 
            pt_indices = np.random.choice(
                pcl.shape[0], points_per_object, replace=False)
            np.random.shuffle(pt_indices)
            pcl = pcl[pt_indices]  # n by 3
            
        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        return result
    
    def colormap(x, y, z):
        vec = np.array([x, y, z])
        vec = np.clip(vec, 0.001, 1.0)
        norm = np.sqrt(np.sum(vec**2))
        vec /= norm
        return [vec[0], vec[1], vec[2]]

    xml_head = \
    f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{camera_extrinsic['location']}" target="{camera_extrinsic['look_at']}" up="{camera_extrinsic['up']}"/>
            </transform>
            <float name="fov" value="{camera_intrinsic['fov']}"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{image_size['xres']}"/>
                <integer name="height" value="{image_size['yres']}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """
    
    xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """
    
    xml_tail = \
    f"""
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-0.2"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="{light_extrinsic['location']}" target="{light_extrinsic['look_at']}" up="{light_extrinsic['up']}"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """

    pcl = pts
    pcl = standardize_bbox(pcl, 4096)
    
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125
    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0]+0.5, pcl[i, 1]+0.5, pcl[i, 2]+0.5-0.0125)
        xml_segments.append(xml_ball_segment.format(
            pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    
    
    # for i in range(pts.shape[0]):
    #     xml_segments.append(xml_ball_segment.format(
    #         pts[i, 0], pts[i, 1], pts[i, 2], *pts[i, 3:]/255))    
    
    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)

    with open('./utils/visualize_tmp/mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)


def test_mitsuba():
    import cv2
    import mitsuba as mi
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    
    mi.set_variant('scalar_rgb')

    pts_path = './utils/visualize_tmp/bowl.txt'
    save_dir = './utils/visualize_tmp/bowl'
    exists_or_mkdir(save_dir)
    
    # pts = np.load(pts_path)[:, :6]
    pts = np.loadtxt(pts_path)
    theta = 0 / 180 * np.pi
    rot_x_180 = np.array([[ np.cos(theta), np.sin(theta), 0],
                          [-np.sin(theta), np.cos(theta), 0],
                          [ 0            , 0            , 1]])
    pts = (rot_x_180 @ pts.T).T
    print(pts.shape)
    image_size={'xres':640, 'yres':640}
    render_num = 12
    r = 4
    z = 4
    x = [r * np.cos(i * np.pi / render_num * 2) for i in range(render_num)]
    y = [r * np.sin(i * np.pi / render_num * 2) for i in range(render_num)]
    for i in range(render_num):
        if i != 8:
            continue
        camera_extrinsic={'location': f"{x[i]},{y[i]},{z}", 'look_at': "0,0,0", 'up': "0,0,1"}
        light_extrinsic={'location': "0.001,0.001,20", 'look_at': "0,0,0", 'up': "0,0,1"}
        generate_xml_for_mitsuba(pts=pts, image_size=image_size, camera_extrinsic=camera_extrinsic, light_extrinsic=light_extrinsic)
        # Absolute or relative path to the XML file
        filename = './utils/visualize_tmp/mitsuba_scene.xml'

        # Load the scene for an XML file
        scene = mi.load_file(filename)
        img = mi.render(scene)
        
        # Write the rendered image to an EXR file
        exr_path = os.path.join(save_dir, f'{str(i)}.exr')
        mi.Bitmap(img).write(exr_path)
        
        # png_path = exr_path.replace('exr', 'png')
        # img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED) * 255
        # cv2.imwrite(png_path, img)

    
if __name__ == '__main__':
    test_mitsuba()
    
    
    
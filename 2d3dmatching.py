from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import os
from tqdm import tqdm
import open3d as o3d

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

SELF_IMPLEMENT = False
ITERATIONS = 100

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model, cameraMatrix, distCoeffs=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    # TODO: solve PnP problem using OpenCV
    # Hint: you may use "Descriptors Matching and ratio test" first
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_query, desc_model, k=2)

    # Lowe's ratio test 過濾
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    kp_2D = np.float32([kp_query[m.queryIdx] for m in good_matches])
    kp_3D = np.float32([kp_model[m.trainIdx] for m in good_matches])

    if SELF_IMPLEMENT:
        from my_solvePnPRansac import my_solvePnPRansac
        image_width, image_height = 1080, 1920
        reproj_error = 0.01 * max(image_width, image_height)
        success, rvec, tvec, inliers = my_solvePnPRansac( # cv2.solvePnPRansac
            kp_3D, kp_2D, cameraMatrix, distCoeffs, ITERATIONS, reproj_error
        )
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            kp_3D, kp_2D, cameraMatrix, distCoeffs, iterationsCount = ITERATIONS
        )

    return success, rvec, tvec, inliers

def rotation_error(rvec, rotq_gt):
    #TODO: calculate rotation error
    R_est, _ = cv2.Rodrigues(rvec)

    # Convert the Rotation object to a 3x3 rotation matrix
    R_gt = R.from_quat(rotq_gt).as_matrix()

    # relative rotation
    R_err = R_gt @ R_est.T

    # rotation angle (rad)
    theta = R.from_matrix(R_err).magnitude()

    return theta

def translation_error(tvec, tvec_gt):
    #TODO: calculate translation error
    return np.linalg.norm(tvec - tvec_gt)

def normal_from_points(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    u = p2 - p1
    v = p3 - p1
    n = np.cross(u, v)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("The three points are collinear and the plane normal vector cannot be defined")
    return - n / n_norm   # 回傳單位法向量

def get_camera_center(rvec, tvec):
    R_c_w = R.from_rotvec(rvec).as_matrix()
    t = np.array(tvec)
    C = -R_c_w.T @ t
    return C.ravel() # (3,1)-> (3,)

    # method 2
    # P = cameraMatrix @ np.concatenate((R, t.reshape(3, 1)), axis=1)
    #  SVD: last column of V (or Vt.T[:,-1]) is nullspace
    # U, S, Vt = np.linalg.svd(P)
    # Ch = Vt[-1]         # homogeneous 4-vector
    # Ch = Ch / Ch[-1]    # dehomogenize
    # return Ch[:3]

def create_simple_camera_pyramid(rvec, tvec, inv_cameraMatrix, camera_size, scale=0.1, color=[1.0, 0.0, 0.0]):
    """
    Args:
        rvec (np.ndarray): 旋轉向量 (3,)。
        tvec (np.ndarray): 平移向量 (3,)。
        camera_size: (w, h)
        scale (float): 金字塔的大小，控制遠端平面的尺寸和深度。
        color (list): 金字塔的顏色 (RGB)。

    Returns:
        camera_center
        o3d.geometry.LineSet: 表示相機金字塔邊緣的 LineSet 物件。
    """
    
    # 1. 在相機座標系 (Camera Frame) 中定義金字塔的頂點
    # 假設相機位於原點 (0, 0, 0)，面向 +Z 軸 (Colmap 通常是 Z 軸向外)
    # 頂點編號：
    # 0: 相機中心 C (0, 0, 0)
    # 1-4: 遠端平面的四個角點 P1, P2, P3, P4
    
    # 金字塔的遠端深度/長度
    depth = 1 * scale
    
    # 遠端平面的半寬和半高
    half_width = camera_size[0] * 0.5 * scale
    half_height = camera_size[1] * 0.5 * scale
    
    points_c = np.array([                    
        [-half_width, -half_height, depth],       # 1: P1 (左下)
        [ half_width, -half_height, depth],       # 2: P2 (右下)
        [ half_width,  half_height, depth],       # 3: P3 (右上)
        [-half_width,  half_height, depth],       # 4: P4 (左上)
    ], dtype=np.float64)

    # 2. 構建相機到世界的轉換矩陣 T_c_w
    R_c_w = R.from_rotvec(rvec.flatten()).as_matrix()
    points_w = ( R_c_w.T @ ( inv_cameraMatrix @ points_c.T - tvec.reshape(3, 1) ) ).T
    camera_center = get_camera_center(rvec, tvec).reshape(1, 3)

    center2base = points_w - camera_center
    center2base_length = np.linalg.norm(center2base, axis = 1).reshape(4, 1)
    points_w = camera_center + center2base / center2base_length * np.mean(center2base_length)
    
    points_w = np.concatenate((points_w, camera_center), axis=0)

    lines = [ 
        [0, 1], [1, 2], [2, 3], [3, 0], # 遠端平面的邊緣 (四邊形)
        [0, 4], [1, 4], [2, 4], [3, 4]  # 連接中心到四個角點 (金字塔的邊)
    ]

    # 創建 LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_w),
        lines=o3d.utility.Vector2iVector(lines)
    )
    
    # 設置顏色
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return camera_center.reshape(-1), line_set

def get_point_cloud(points3D_df):
    #TODO: visualize the camera pose
    # 取出 XYZ 和 RGB
    xyz = np.vstack(points3D_df["XYZ"].to_numpy())
    rgb = np.vstack(points3D_df["RGB"].to_numpy()) / 255.0  # normalize to [0,1]

    # 3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def visualization(GeometryList):
    #TODO: visualize the camera pose
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 加入幾何體 (例如 LineSet, PointCloud 等)
    for obj in GeometryList:
        vis.add_geometry(obj)
    # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
    vis.run()
    # 關閉視窗
    vis.destroy_window()
    return

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    if SELF_IMPLEMENT:
        camera_poses_path = 'est_camera_poses_myRANSAC.pkl'
    else:
        camera_poses_path = 'est_camera_poses.pkl'

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    IMAGE_ID_LIST = images_df['IMAGE_ID'][np.where(images_df['NAME'].str.startswith('valid_img'))[0]].to_list()
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    fname_list = []
    if os.path.exists(camera_poses_path):
        camera_poses_df = pd.read_pickle(camera_poses_path)
        r_list = camera_poses_df['rvec'].to_list()
        t_list = camera_poses_df['tvec'].to_list()
        rotation_error_list = camera_poses_df['rotation_error'].to_list()
        translation_error_list = camera_poses_df['translation_error'].to_list()
    else:
        for idx in tqdm(IMAGE_ID_LIST):
            # Load quaery image
            fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
            fname_list.append(fname)
            # rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

            # Find correspondance and solve pnp
            retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model),
                                                    cameraMatrix, distCoeffs)
            # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
            tvec = tvec.reshape(-1)
            rvec = rvec.reshape(-1)
            r_list.append(rvec.tolist())
            t_list.append(tvec.tolist())

            # Get camera pose groudtruth
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values

            # Calculate error
            r_error = rotation_error(rvec, rotq_gt)
            t_error = translation_error(tvec, tvec_gt)
            rotation_error_list.append(r_error)
            translation_error_list.append(t_error)
        
        camera_poses_df = pd.DataFrame({"IMAGE_ID": IMAGE_ID_LIST,
                                        "NAME": fname_list,
                                        "rvec": r_list,
                                        "tvec": t_list,
                                        "rotation_error": rotation_error_list,
                                        "translation_error": translation_error_list
                                        })
        camera_poses_df.to_pickle(camera_poses_path)
    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    # Median
    translation_error_median = np.median(translation_error_list)
    rotation_error_median = np.median(rotation_error_list)

    print("Translation Error Median:", translation_error_median)
    print("Rotation Error Median (rad):", rotation_error_median)

    # TODO: result visualization
    # To draw trajectory, sort the results by validation image number
    val_img_name = images_df['NAME'][np.where(images_df['NAME'].str.startswith('valid_img'))[0]]
    val_img_num = val_img_name.str.extract(r'(\d+)').astype(int).to_numpy().squeeze()
    sort_idx = np.argsort(val_img_num)
    sorted_r_list = np.array(r_list)[sort_idx]
    sorted_t_list = np.array(t_list)[sort_idx]
    inv_cameraMatrix = np.linalg.inv(cameraMatrix)

    pyramids = []
    camera_centers = []
    for r, t in zip(sorted_r_list, sorted_t_list):
        # TODO: calculate camera pose in world coordinate system
        camera_center, line_set = create_simple_camera_pyramid(np.array(r), np.array(t),
                                                               inv_cameraMatrix,
                                                               camera_size=(1080, 1920),
                                                               scale = 0.15, # camera_size * scale
                                                               color=[1.0, 0.0, 0.0] # red
                                                               )
        pyramids.append(line_set)
        camera_centers.append(camera_center)
    
    # point cloud
    pcd = get_point_cloud(points3D_df)
    # trajectory
    trajectory = np.vstack([np.arange(len(pyramids)-1), np.arange(len(pyramids)-1) + 1]).T
    trajectory_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(camera_centers),
        lines=o3d.utility.Vector2iVector(trajectory)
    )
    trajectory_line.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * len(trajectory)) # green
    
    visualization([pcd, trajectory_line] + pyramids)
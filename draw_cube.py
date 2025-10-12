import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import cv2
import glob

def generate_face_points(v0, v1, v2, v3, n):
    """
    Given a quadrilateral with four vertices (v0, v1, v2, v3), 
    generate an n×n uniform grid of points
    v0-v1-v2-v3 in clockwise or counterclockwise order.
    """
    grid = []
    for i in range(n):
        for j in range(n):
            u = i / (n - 1)
            v = j / (n - 1)
            # 雙線性插值
            p = (1 - u) * (1 - v) * v0 + u * (1 - v) * v1 + u * v * v2 + (1 - u) * v * v3
            grid.append(p)
    return np.array(grid)

def generate_cube_points(cube_vertices, n):
    """
    由 8 頂點產生 cube 的 6 個面上的 n×n 點
    cube_vertices 順序需符合 create_box 的輸出
    """
    faces = [
        [0, 1, 3, 2],  # bottom (y=0 plane)
        [4, 5, 7, 6],  # top    (y=1 plane)
        [0, 1, 5, 4],  # front  (z=0 plane)
        [2, 3, 7, 6],  # back   (z=1 plane)
        [1, 3, 7, 5],  # right  (x=1 plane)
        [0, 2, 6, 4],  # left   (x=0 plane)
    ]
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 0, 255),  # magenta
    ]
    all_points, all_colors = [], []
    for face_id, idx in enumerate(faces):
        v0, v1, v2, v3 = [cube_vertices[i] for i in idx]
        pts = generate_face_points(v0, v1, v2, v3, n)
        all_points.append(pts)
        all_colors.append(np.tile(colors[face_id], (pts.shape[0], 1)))
    return np.vstack(all_points), np.vstack(all_colors)

def my_projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs=None):
    # Rodrigues: rvec -> R
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # 3D -> Camera coordinates
    points3D = np.asarray(points3D, dtype=np.float32).reshape(-1, 3).T  # (3, N)
    cam_points = R @ points3D + t  # (3, N)

    x = cam_points[0, :] / cam_points[2, :]
    y = cam_points[1, :] / cam_points[2, :]

    if distCoeffs is not None:
        distCoeffs = distCoeffs.ravel().tolist()
        if len(distCoeffs) >= 4:
            k1, k2, p1, p2 = distCoeffs[:4]
        if len(distCoeffs) >= 5:
            k3 = distCoeffs[4]
        else:
            k3 = 0.0
        r2 = x**2 + y**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        x_radial = x * radial
        y_radial = y * radial

        x_tang = 2*p1*x*y + p2*(r2 + 2*x**2)
        y_tang = p1*(r2 + 2*y**2) + 2*p2*x*y

        x = x_radial + x_tang
        y = y_radial + y_tang

    fx, fy = cameraMatrix[0,0], cameraMatrix[1,1]
    cx, cy = cameraMatrix[0,2], cameraMatrix[1,2]

    u = fx * x + cx
    v = fy * y + cy

    imgpts = np.vstack((u, v)).T  # (N, 2)
    return imgpts

def draw_cube_on_image(img, cube_vertices, cameraMatrix, distCoeffs, rvec, tvec, n=10):
    # 生成 cube 的面點
    points3D, colors = generate_cube_points(cube_vertices, n)

    # 投影到影像平面
    imgpts, _ = cv2.projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs)
    imgpts = imgpts.reshape(-1, 2)
    # imgpts = my_projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs)

    # 計算每個點的深度
    R_c_w = R.from_rotvec(rvec).as_matrix()
    t = np.array(tvec)
    camera_center = -R_c_w.T @ t
    depths = np.linalg.norm(points3D - camera_center, axis=1)

    # 按深度從遠到近排序
    sort_idx = np.argsort(-depths)  # 遠到近
    imgpts = imgpts[sort_idx]
    colors = colors[sort_idx]

    # 畫點
    for (x, y), color in zip(imgpts.astype(int), colors):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 5, color.tolist(), -1)

    return img

if __name__ == "__main__":
    save_fold = 'painted_images'
    os.makedirs(save_fold, exist_ok=True)
    cube_vertices = np.load('cube_vertices.npy')
    camera_poses_path = 'est_camera_poses.pkl'
    if not os.path.exists(camera_poses_path):
        camera_poses_path = 'est_camera_poses_myRANSAC.pkl'
    camera_poses_df = pd.read_pickle(camera_poses_path)
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    for idx in tqdm(camera_poses_df.index):
        img_name = camera_poses_df.iloc[idx]['NAME']
        img = cv2.imread(os.path.join('data', 'frames', img_name))
        rvec = np.array(camera_poses_df.iloc[idx]['rvec'])
        tvec = np.array(camera_poses_df.iloc[idx]['tvec'])
        # rvec = R.from_quat(true_quaternion[idx]).as_rotvec()
        # tvec = true_t[idx]
        
        painted_img = draw_cube_on_image(img, cube_vertices, cameraMatrix,
                                         distCoeffs, rvec, tvec, n=20)
        
        cv2.imwrite(os.path.join(save_fold, img_name), painted_img)
    
    output_video = "draw_cube.mp4"
    images = glob.glob(os.path.join(save_fold, "*.jpg"))

    images_df = pd.read_pickle("data/images.pkl")
    val_img_name = images_df['NAME'][np.where(images_df['NAME'].str.startswith('valid_img'))[0]]
    val_img_num = val_img_name.str.extract(r'(\d+)').astype(int).to_numpy().squeeze()
    sort_idx = np.argsort(val_img_num)
    images = val_img_name.iloc[sort_idx].to_list()

    if images is None:
        raise ValueError(f"No .jpg in folder {save_fold}")
    
    frame = cv2.imread(os.path.join(save_fold, images[0]))
    height, width, layers = frame.shape

    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可改為 'mp4v' 輸出 mp4
    fps = 10  # 每秒幾張
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 把所有圖片寫進影片
    for img_path in images:
        img = cv2.imread(os.path.join(save_fold, img_path))
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
import numpy as np
import cv2

def cosine_similarity(v1, v2):
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def trilateration(p1, r1, p2, r2, p3, r3):
    """
    三球交會解 (可能有兩個解)
    p1, p2, p3: 已知點 (x, y, z)
    r1, r2, r3: 對應距離
    return: [sol1, sol2] 兩個解 (np.array)
    """

    P1 = np.array(p1, dtype=float)
    P2 = np.array(p2, dtype=float)
    P3 = np.array(p3, dtype=float)

    # 單位向量 ex
    ex = (P2 - P1) / np.linalg.norm(P2 - P1)
    i = np.dot(ex, P3 - P1)

    # 單位向量 ey
    temp = P3 - P1 - i * ex
    ey = temp / np.linalg.norm(temp)

    # 單位向量 ez
    ez = np.cross(ex, ey)

    d = np.linalg.norm(P2 - P1)
    j = np.dot(ey, P3 - P1)

    # 坐標系下的 x, y
    x = (r1**2 - r2**2 + d**2) / (2*d)
    y = (r1**2 - r3**2 + i**2 + j**2 - 2*i*x) / (2*j)

    # z 可能 ±
    z_sq = r1**2 - x**2 - y**2
    if z_sq < 0:
        raise ValueError("三個球沒有共同交點")
    z = np.sqrt(z_sq)

    # 轉換回全域座標
    sol1 = P1 + x*ex + y*ey + z*ez
    sol2 = P1 + x*ex + y*ey - z*ez

    return sol1, sol2

def my_solveP3P(pts3D, pts2D, K):
    """
    pts3D: (N, 3)
    pts2D: (N, 2)
    K: (3, 3) Camera Intrinsic Parameters:
    """
    assert pts3D.shape[0] == pts2D.shape[0] == 3, "P3P needs 3 pairs"

    pts2D_homogeneous = np.hstack((pts2D, np.ones((pts2D.shape[0], 1))))

    v = np.linalg.inv(K) @ pts2D_homogeneous.T
    Cab = cosine_similarity(v[0], v[1])
    Cac = cosine_similarity(v[0], v[2])
    Cbc = cosine_similarity(v[1], v[2])

    Rab = np.linalg.norm(pts3D[0] - pts3D[1])
    Rac = np.linalg.norm(pts3D[0] - pts3D[2])
    Rbc = np.linalg.norm(pts3D[1] - pts3D[2])

    K1 = (Rbc / Rac) **2
    K2 = (Rbc / Rab) **2
    K1K2 = K1 * K2
    K1_K2_mm = K1K2 - K1 - K2
    K1_K2_mp = K1K2 - K1 + K2
    K1_K2_pm = K1K2 + K1 - K2

    G4 = K1_K2_mm**2 - 4 * K1K2 * Cbc**2

    term_A = 4 * K1_K2_mm * K2 * (1 - K1) * Cab
    term_B = 4 * K1 * Cbc * (K1_K2_mp * Cac + 2 * K2 * Cab * Cbc)
    G3 = term_A + term_B

    term_C = (2 * K2 * (1 - K1) * Cab)**2
    term_D = 2 * K1_K2_mm * K1_K2_pm
    term_E_inner = ((K1 - K2) * Cbc**2 + K1 * (1 - K2) * Cac**2 
                    - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
    term_E = 4 * K1 * term_E_inner
    G2 = term_C + term_D + term_E

    term_F = 4 * K1_K2_pm * K2 * (1 - K1) * Cab
    term_G_inner = (K1_K2_mp * Cac * Cbc + 2 * K1K2 * Cab * Cac**2)
    term_G = 4 * K1 * term_G_inner
    G1 = term_F + term_G

    G0 = K1_K2_pm**2 - 4 * K1**2 * K2 * Cac**2

    x = np.roots([G4, G3, G2, G1, G0])
    a_sq = Rab**2 / (1 + x**2 - 2*x*Cab)
    a = np.sqrt(a_sq[a_sq >= 0])
    x = x[a_sq >= 0] # remove impossable value

    m = 1 - K1
    p = 2 * (K1 * Cac - x * Cbc)
    q = x ** 2 - K1
    m2 = 1
    p2 = 2 * (- x * Cbc)
    q2 = x ** 2 * (1 - K2) + 2 * x * K2 * Cab - K2
    y1 = -(m2 * q - m * q2) / (p * m2 - p2 * m)
    y2 = -(p2 * q - p * q2) / (m2 * q - m * q2)
    y = np.mean((y1, y2), axis=0)

    b = x * a
    c = y * a

    sols = []
    for i in range(len(a)):
        sol1, sol2 = trilateration(pts3D[0], a[i], pts3D[1], b[i], pts3D[2], c[i])
        sols.append(sol1)
        sols.append(sol2)
    sols = np.stack(sols)

    abs_lambda = []
    for i in range(3):
        abs_lambda.append(np.linalg.norm(pts3D[i] - sols, axis=1) / np.linalg.norm(pts2D[i]))
    abs_lambda = np.stack(abs_lambda)




    N = pts3D.shape[0]
    pts3D_h = np.hstack([pts3D, np.ones((N,1))])  # (N,4)

    A = []
    for i in range(N):
        X = pts3D_h[i]
        u, v = pts2D[i]

        # 對應 2 行
        A.append(np.hstack([np.zeros(4), -X, v*X]))
        A.append(np.hstack([X, np.zeros(4), -u*X]))
    A = np.array(A)

    # SVD 解
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3,4)

    # 分解 P 成 K [R|t]
    M = np.linalg.inv(K) @ P
    R, t = M[:,:3], M[:,3]

    # 強制 R 為正交矩陣
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # 轉換為 rvec (Rodrigues)
    rvec, _ = cv2.Rodrigues(R)

    return True, rvec, t

def my_solvePnPRansac(kp_3D, kp_2D, cameraMatrix, distCoeffs=None,
                      iterations=100, reproj_error=8.0, min_points=6):

    kp_3D = np.asarray(kp_3D, dtype=np.float32)
    kp_2D = np.asarray(kp_2D, dtype=np.float32)
    N = len(kp_3D)

    best_inliers = []
    best_rvec, best_tvec = None, None

    for _ in range(iterations):
        # 1. 隨機選取最少點數
        idx = np.random.choice(N, min_points, replace=False)
        subset_3D, subset_2D = kp_3D[idx], kp_2D[idx]

        # 2. 嘗試解 PnP
        success, rvec, tvec = cv2.solvePnP(
            subset_3D, subset_2D, cameraMatrix, distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            continue

        # 3. 投影所有點
        proj_points, _ = cv2.projectPoints(kp_3D, rvec, tvec, cameraMatrix, distCoeffs)

        proj_points = proj_points.squeeze()
        errors = np.linalg.norm(kp_2D - proj_points, axis=1)

        # 4. 找到 inliers
        inliers = np.where(errors < reproj_error)[0]

        # 5. 更新最佳解
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_rvec, best_tvec = rvec, tvec

    # 用最佳 inliers 再精煉一次
    if len(best_inliers) >= min_points:
        success, best_rvec, best_tvec = cv2.solvePnP(
            kp_3D[best_inliers], kp_2D[best_inliers],
            cameraMatrix, distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return True, best_rvec, best_tvec, best_inliers
    else:
        return False, None, None, []

import open3d as o3d
import numpy as np

def fit_plane_ransac(point_array, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    '''
    使用 RANSAC 拟合空间平面并返回平面参数、内点和外点

    参数:
    - point_array: np.ndarray, shape (N, 3)，输入的点云坐标
    - distance_threshold: float，内点的最大距离阈值
    - ransac_n: int，RANSAC 每次拟合所需的最小样本数（拟合平面为3）
    - num_iterations: int，RANSAC 最大迭代次数

    返回:
    - plane_model: list[float], [a, b, c, d] 平面方程 ax + by + cz + d = 0
    - inliers: np.ndarray, shape (M, 3)，拟合平面上的内点
    - outliers: np.ndarray, shape (N-M, 3)，不在平面上的外点
    '''

    # 转为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)

    # RANSAC 拟合平面
    plane_model, inlier_idxs = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    # 提取内点和外点
    inlier_mask = np.zeros(len(point_array), dtype=bool)
    inlier_mask[inlier_idxs] = True
    inliers = point_array[inlier_mask]
    outliers = point_array[~inlier_mask]

    return plane_model, inliers, outliers


# 构造一个带噪平面点云
np.random.seed(42)
plane_pts = np.random.rand(100, 2)
z = 0.5 * plane_pts[:, 0] + 0.3 * plane_pts[:, 1] + 0.1  # z = 0.5x + 0.3y + 0.1
points = np.hstack((plane_pts, z[:, None])) + np.random.normal(scale=0.01, size=(100, 3))

# 添加一些离群点
outliers = np.random.uniform(-1, 1, size=(30, 3))
points_all = np.vstack((points, outliers))

# 执行RANSAC拟合
plane_model, inliers, outliers = fit_plane_ransac(points_all, distance_threshold=0.02)

print("拟合平面参数: ax + by + cz + d = 0")
print(f"  a={plane_model[0]:.3f}, b={plane_model[1]:.3f}, c={plane_model[2]:.3f}, d={plane_model[3]:.3f}")
print(f"内点数: {len(inliers)}, 外点数: {len(outliers)}")

# 可视化
pcd_in = o3d.geometry.PointCloud()
pcd_in.points = o3d.utility.Vector3dVector(inliers)
pcd_in.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色内点

pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(outliers)
pcd_out.paint_uniform_color([1.0, 0.0, 0.0])  # 红色外点

o3d.visualization.draw_geometries([pcd_in, pcd_out])

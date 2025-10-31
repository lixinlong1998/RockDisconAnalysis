import numpy as np
import open3d as o3d
from matplotlib import cm  # 用于 colormap

# --- 1. 构建正二十面体 ---
phi = (1 + np.sqrt(5)) / 2
vertices = np.array([
    [-1, phi, 0],
    [1, phi, 0],
    [-1, -phi, 0],
    [1, -phi, 0],
    [0, -1, phi],
    [0, 1, phi],
    [0, -1, -phi],
    [0, 1, -phi],
    [phi, 0, -1],
    [phi, 0, 1],
    [-phi, 0, -1],
    [-phi, 0, 1],
], dtype=np.float64)
vertices /= np.linalg.norm(vertices[0])

triangles = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
], dtype=np.int32)

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(triangles)
)
mesh.compute_vertex_normals()

# --- 2. 从三角面采样点云 ---
num_points = 50000
pcd = mesh.sample_points_uniformly(number_of_points=num_points)

# --- 3. 添加高斯噪声（沿法向） ---
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

noise_strength = 0.02  # 控制噪声强度
gaussian_noise = np.random.normal(scale=noise_strength, size=points.shape[0])
points_noisy = points + (normals.T * gaussian_noise).T
pcd.points = o3d.utility.Vector3dVector(points_noisy)

# --- 4. 颜色映射 ---
# 1. 归一化噪声值到 0~1 区间（用于颜色映射）
noise_abs = np.abs(gaussian_noise)  # 正数表示扰动强度
noise_norm = (noise_abs - noise_abs.min()) / (noise_abs.max() - noise_abs.min() + 1e-8)
# 2. 使用 matplotlib 的 colormap（例如 'jet'）将标量映射为 RGB
colormap = cm.get_cmap('jet')  # 你也可以用 'viridis', 'plasma', 'hot' 等
colors = colormap(noise_norm)[:, :3]  # 丢弃 alpha 通道，保留 RGB
# 3. 设置点云颜色
pcd.colors = o3d.utility.Vector3dVector(colors)

# --- 5. 可视化 ---
o3d.visualization.draw_geometries([pcd], point_show_normal=False)

# --- 6. 保存为 .ply 和 .obj ---
o3d.io.write_point_cloud("icosahedron_noisy.ply", pcd)
o3d.io.write_point_cloud("icosahedron_noisy.obj", pcd)
print("已保存为 'icosahedron_noisy.ply' 和 'icosahedron_noisy.obj'")

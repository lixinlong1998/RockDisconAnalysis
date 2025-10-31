import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
import matplotlib.patches as patches

# 设置随机种子确保可重复
np.random.seed(42)

# 定义6个多边形（注意左上角的为带有空洞的正方形）
polygons = [
    Polygon(shell=[(0, 0), (0, 1), (1, 1), (1, 0)],
            holes=[[(0.2, 0.5), (0.5, 0.8), (0.8, 0.5), (0.5, 0.2)]]),  # 带洞正方形
    Polygon([(0, 0), (0.5, 1), (1, 1), (1.5, 0.5), (1, 0)]),  # 五边形
    Polygon([(0.5, 0), (1, 1), (0, 1)]),  # 正三角形
    Polygon([(0, 0), (0.2, 1), (0.4, 0.4), (1, 0.1)]),  # 凹四边形
    Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # 普通正方形
    Polygon([(0, 0), (1, 0.2), (1.2, 0.3), (0.2, 0.1)])  # 平行四边形/倾斜四边形
]

# 散点密度参数
k = 1000

# 每个多边形中采样点
sampled_points = []
for poly in polygons:
    minx, miny, maxx, maxy = poly.bounds
    points = []
    while len(points) < k:
        x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            points.append((x, y))
    sampled_points.append(np.array(points))

# 绘制图形与采样点
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()
for i, (poly, pts) in enumerate(zip(polygons, sampled_points)):
    x, y = poly.exterior.xy
    axs[i].plot(x, y, color='black')
    for hole in poly.interiors:
        hx, hy = hole.xy
        axs[i].plot(hx, hy, color='black')
    axs[i].scatter(pts[:, 0], pts[:, 1], color='red', s=10)
    axs[i].set_aspect('equal')
plt.tight_layout()
plt.show()

# 保存为“伪三维点” + 法向量 + cluster_point结构以供下一步执行
data = {}  # 所有点 (N, 3)
cluster_point = {}
dip_dict = {}
all_3d_points = []
point_idx_offset = 0

# 给每个 polygon 设置不同法向量
normals = [
    [0, 0, 1],  # z向上
    [0, 1, 1],  # y偏上
    [1, 1, 1],  # 斜向
    [1, 0, 1],  # x偏上
    [1, 1, 0],  # x-y平面内法向
    [0, 1, 0.5]  # y-z偏向
]

for i, pts in enumerate(sampled_points):
    pts_3d = np.hstack([pts, np.zeros((pts.shape[0], 1))])  # 添加z=0
    all_3d_points.append(pts_3d)
    indices = list(range(point_idx_offset, point_idx_offset + len(pts_3d)))
    cluster_point[("poly", i)] = indices
    dip_dict[("poly", i)] = [0, 0, 0] + normals[i]  # 前三项占位，后三项为法向量
    point_idx_offset += len(pts_3d)

# 合并数据
all_3d_points = np.vstack(all_3d_points)

import numpy as np
import pickle
import matplotlib.pyplot as plt


# 从前面步骤中保留的变量
# - all_3d_points: np.ndarray (N, 3)
# - cluster_point: dict of {polygon_id: [indices]}
# - dip_dict: dict of {polygon_id: (A, B, C)}

def project_points_to_plane(points, normal):
    """
    将三维点投影到给定法向量定义的平面上
    返回投影后的二维坐标和平面局部坐标系基向量（u, v）
    """
    normal = normal / np.linalg.norm(normal)
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    projected_2d = np.dot(points, np.vstack([u, v]).T)
    return projected_2d, u, v


def double_farest_search(projected_2d, point_indices):
    center = projected_2d.mean(axis=0)
    dists_to_center = np.linalg.norm(projected_2d - center, axis=1)
    idx1 = np.argmax(dists_to_center)
    dists_to_idx1 = np.linalg.norm(projected_2d - projected_2d[idx1], axis=1)
    idx2 = np.argmax(dists_to_idx1)
    max_len = np.max(dists_to_idx1)
    return max_len, point_indices[idx1], point_indices[idx2]


# 主函数：执行迹线提取
trace_dict = {}
for i, points in enumerate(sampled_points):
    center = points.mean(axis=0)
    dists_to_center = np.linalg.norm(points - center, axis=1)
    idx1 = np.argmax(dists_to_center)
    dists_to_idx1 = np.linalg.norm(points - points[idx1], axis=1)
    idx2 = np.argmax(dists_to_idx1)
    max_len = np.max(dists_to_idx1)
    points_coord_start = points[idx1, :]
    points_coord_end = points[idx2, :]
    trace_dict[i] = [points_coord_start, points_coord_end]

# 可视化结果：散点图 + 迹线
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs = axs.flatten()

for i, points in enumerate(sampled_points):
    axs[i].scatter(points[:, 0], points[:, 1], s=8, c='red', label='Projected Points')

    # 画迹线
    projected_pts = np.asarray(trace_dict[i])
    axs[i].plot(projected_pts[:, 0], projected_pts[:, 1], 'b-', linewidth=2, label='Trace Line')

    axs[i].set_title(f'Polygon {i}')
    axs[i].axis('equal')
    axs[i].legend()

plt.tight_layout()
plt.show()

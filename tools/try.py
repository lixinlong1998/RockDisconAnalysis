import open3d as o3d
import numpy as np

# # 1. 读取点云（PLY 格式）
# ply_file = r"D:\Research\20250313_RockFractureSeg\Experiments\Exp1_DiscontinuityExtraction\G3033_9_Part5\G3033_9_Part5 XYZ-HSV-early_classification.ply"  # 修改为你的PLY文件路径
# pcd = o3d.io.read_point_cloud(ply_file)
#
# # 2. 检查是否已有法向，如果没有则计算
# if not pcd.has_normals():
#     print("未检测到法向，正在计算...")
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
#     # 可选：调整法向方向
#     pcd.orient_normals_consistent_tangent_plane(k=30)
#
#
# # 3. 创建法向箭头的线段可视化
# def create_normal_lines(pcd, scale=0.02):
#     points = np.asarray(pcd.points)
#     normals = np.asarray(pcd.normals)
#     lines = []
#     colors = []
#     line_points = []
#
#     for i in range(len(points)):
#         start = points[i]
#         end = start + normals[i] * scale
#         line_points.append(start)
#         line_points.append(end)
#         lines.append([2 * i, 2 * i + 1])
#         colors.append([1, 0, 0])  # 红色箭头
#
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(line_points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(colors)
#     return line_set
#
#
# normal_lines = create_normal_lines(pcd, scale=0.05)
# print('here')
# # 4. 可视化点云 + 法向箭头
# o3d.visualization.draw_geometries([pcd, normal_lines],
#                                   point_show_normal=False,
#                                   window_name='点云法向可视化',
#                                   width=800, height=600)
import math

batch_size = math.ceil(100 / 10)
print(batch_size)
print([(i, i + batch_size) for i in range(0, 100, batch_size)])

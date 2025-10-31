import open3d as o3d
import numpy as np

# 使用示例
ply_file_path = r'D:\Research\20250313_RockFractureSeg\Experiments\Exp1_DiscontinuityExtraction\G3033_9_Part2 XYZ-HSV-early_classification.ply'
# 读取PLY文件
pcd = o3d.io.read_point_cloud(ply_file_path)

# 检查数据完整性
assert len(pcd.points) > 0, "文件无顶点数据！"

# 获取第203个顶点
index = 202
vertex_pos = np.asarray(pcd.points)[index]
vertex_color = np.asarray(pcd.colors)[index] * 255 if pcd.has_colors() else None

print(f"顶点位置: {vertex_pos}")
print(f"顶点颜色: {vertex_color}")

# 查看附近几个点颜色是否一致，不太一致
for index in range(203,210):
    vertex_pos = np.asarray(pcd.points)[index]
    vertex_color = np.asarray(pcd.colors)[index] * 255 if pcd.has_colors() else None
    print(f"顶点位置: {vertex_pos}")
    print(f"顶点颜色: {vertex_color}")


# 可视化
# o3d.visualization.draw_geometries([pcd])

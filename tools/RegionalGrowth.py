import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt  # 添加这行导入


def region_growing_segmentation(pcd, min_cluster_size=100, angle_threshold=30, curvature_threshold=0.05):
    """
    区域生长法分割点云中的平面结构

    参数:
        pcd: 输入点云(open3d.geometry.PointCloud)
        min_cluster_size: 最小聚类点数
        angle_threshold: 法线角度阈值(度)
        curvature_threshold: 曲率阈值

    返回:
        分割后的平面列表
    """
    # 估计法线(如果尚未计算)
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

    # 区域生长分割 (使用DBSCAN算法)
    print("Performing clustering...")
    labels = np.array(pcd.cluster_dbscan(eps=0.2,
                                         min_points=50,
                                         print_progress=True))

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    # 为每个聚类分配颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 噪声点设为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 提取每个平面
    planes = []
    for label in range(0, max_label + 1):
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) > min_cluster_size:
            plane = pcd.select_by_index(cluster_indices)
            planes.append(plane)
            print(f"Cluster {label} has {len(cluster_indices)} points")

    return planes


# 加载点云文件
def load_point_cloud(file_path):
    if file_path.endswith('.las') or file_path.endswith('.laz'):
        import laspy
        print(f"Loading LAS file: {file_path}")
        las = laspy.read(file_path)
        print(las)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        print(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    elif file_path.endswith('.ply'):
        print(f"Loading PLY file: {file_path}")
        return o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError("Unsupported file format")


# 主程序
if __name__ == "__main__":
    # 替换为你的点云文件路径
    file_path = r"F:\20240731_UAV_G3033\Aprojects\9\G3033_9_Part1.las"  # 或 "rock_surface.ply"

    try:
        # 加载点云
        pcd = load_point_cloud(file_path)

        print(f"Loaded point cloud with {len(pcd.points)} points")
        print(pcd)

        # 预处理: 去噪和下采样
        print("Downsampling...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        print(f"After Downsampling: {len(pcd.points)} points")

        print("Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"After preprocessing: {len(pcd.points)} points")

        # 区域生长分割
        planes = region_growing_segmentation(pcd)

        # 可视化结果
        print("Visualizing results...")
        # o3d.visualization.draw_geometries(planes)

        # 保存结果(可选)
        print(len(planes))
        for i, plane in enumerate(planes):
            output_path = f"plane_{i}.ply"
            o3d.io.write_point_cloud(output_path, plane)
            print(f"Saved {output_path}")

    except Exception as e:
        print(f"Error: {e}")
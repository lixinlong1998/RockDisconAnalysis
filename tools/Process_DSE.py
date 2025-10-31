# -*- coding: utf-8 -*-
"""
写在前面：这个效果不好，把点云都删得全是空洞
点云预处理流程 (适用于 PLY 格式点云，含颜色和法向)
==================================================
功能步骤：
1) 读取 PLY 格式点云；
2) 去除离群点（SOR + ROR）；
3) 体素下采样，统一点密度；
4) 法向量重新估计与方向一致化；
5) 保存预处理后的点云。

输入数据格式：
    PLY 文件，包含：
        - x, y, z  (float)   三维坐标
        - nx, ny, nz (float) 法向分量
        - red, green, blue (uchar) 颜色

输出：
    - cleaned_pointcloud.ply (预处理后点云)
"""

import open3d as o3d
import numpy as np

def preprocess_pointcloud(input_ply: str,
                          voxel_size: float = 0.005,
                          nb_neighbors: int = 30,
                          std_ratio: float = 2.0,
                          ror_nb_points: int = 12,
                          ror_radius: float = 0.05,
                          output_ply: str = "cleaned_pointcloud.ply"):
    """
    点云预处理主函数

    参数
    ----------
    input_ply : str
        输入的 PLY 文件路径
    voxel_size : float
        下采样体素大小 (m)，根据点云密度选择，一般 0.01 ~ 0.05
    nb_neighbors : int
        统计滤波邻域点数 (Statistical Outlier Removal)
    std_ratio : float
        SOR 标准差阈值，越小越严格
    ror_nb_points : int
        半径滤波最少邻居点数
    ror_radius : float
        半径滤波的搜索半径 (m)
    output_ply : str
        输出清理后的 PLY 文件路径

    返回
    ----------
    o3d.geometry.PointCloud
        预处理后的点云对象
    """

    # 1. 读取点云
    print(">>> 正在读取点云: ", input_ply)
    pcd = o3d.io.read_point_cloud(input_ply)

    print("初始点数:", np.asarray(pcd.points).shape[0])

    # 2. 统计离群点滤波 (SOR)
    print(">>> 统计滤波 (SOR)...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                              std_ratio=std_ratio)

    # 3. 半径离群点滤波 (ROR)
    print(">>> 半径滤波 (ROR)...")
    pcd, ind = pcd.remove_radius_outlier(nb_points=ror_nb_points,
                                         radius=ror_radius)

    print("滤波后点数:", np.asarray(pcd.points).shape[0])

    # 4. 体素下采样
    print(f">>> 下采样 (voxel_size={voxel_size})...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print("下采样后点数:", np.asarray(pcd.points).shape[0])

    # 5. 法向量重新估计与方向一致化
    print(">>> 法向量重新估计...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # 6. 保存处理后的点云
    o3d.io.write_point_cloud(output_ply, pcd)
    print(">>> 已保存预处理点云:", output_ply)

    return pcd


if __name__ == "__main__":
    input_path_ply =r"D:\Research\20250313_RockFractureSeg\Experiments\Exp1_DiscontinuityExtraction\G3033_9_Part2\G3033_9_Part2.ply"
    output_path_ply =r"D:\Research\20250313_RockFractureSeg\Experiments\Exp1_DiscontinuityExtraction\G3033_9_Part2\G3033_9_Part2_clean.ply"
    # 示例调用
    processed_pcd = preprocess_pointcloud(
        input_ply=input_path_ply,
        output_ply=output_path_ply
    )

    # 可视化
    o3d.visualization.draw_geometries([processed_pcd])

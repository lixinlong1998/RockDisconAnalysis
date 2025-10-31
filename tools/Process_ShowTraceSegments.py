import open3d as o3d
import os


def visualize_ply_with_lines(ply_path):
    """
    可视化包含线段 (element edge) 的 PLY 文件

    参数
    ----
    ply_path : str
        .ply 文件路径
    """
    # 尝试以 LineSet 方式加载
    try:
        geometry = o3d.io.read_line_set(ply_path)
        if len(geometry.lines) == 0:
            print("警告：该 PLY 文件中未找到 edge 元素或 lines 数量为 0。")
        o3d.visualization.draw_geometries([geometry])
    except Exception as e:
        print(f"读取为 line_set 失败：{e}")
        print("尝试读取为 triangle mesh 方式展示点云和可能存在的面...")

        # 退一步以 mesh 方式读取
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if not mesh.has_vertices():
            print("该 PLY 文件中不包含点数据。")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    data_path = r'D:\Research\20250313_RockFractureSeg\Experiments\Exp2_DiscontinuityCharacterization\G3033_9_Part2_ClusterAnalysis xyz-js-c-abcd.txt'
    trace_dict_path_ply = os.path.join(os.path.dirname(data_path), 'G3033_9_Part2_Segments.ply')
    visualize_ply_with_lines(trace_dict_path_ply)

import open3d as o3d
import os


def visualize_ply_disks(ply_file):
    """
    使用 Open3D 可视化生成的圆盘
    """
    mesh = o3d.io.read_triangle_mesh(ply_file)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    data_path = r'D:\Research\20250313_RockFractureSeg\Experiments\Exp2_DiscontinuityCharacterization\G3033_9_Part2_ClusterAnalysis xyz-js-c-abcd.txt'
    disk_dict_path_ply = os.path.join(os.path.dirname(data_path),
                                      'G3033_9_Part2_ClusterAnalysis_DiscontinuityDisks.ply')
    visualize_ply_disks(disk_dict_path_ply)

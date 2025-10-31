# Script 1: script1_triangulation.py
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
import argparse


def main(input_path, output_npz, output_ply):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    print(f"[DEBUG] Loaded point cloud with {len(pts)} points")

    # Compute Delaunay triangulation on XY plane
    tri = Delaunay(pts[:, :2])
    triangles = tri.simplices
    print(f"[DEBUG] Generated {len(triangles)} triangles")

    # Save triangles and points to NPZ for downstream processing
    np.savez(output_npz, triangles=triangles, points=pts)
    print(f"[DEBUG] Saved triangulation data to {output_npz}")

    # Also export binary PLY mesh for visualization in CloudCompare
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_ply, mesh, write_ascii=False)
    print(f"[DEBUG] Exported binary mesh PLY: {output_ply}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True, help='Input point cloud (.ply/.las)')
    # parser.add_argument('--output_npz', default='triangulation_data.npz', help='Output NPZ file')
    # parser.add_argument('--output_ply', default='triangulation_mesh.ply', help='Output binary mesh PLY')
    # args = parser.parse_args()
    # main(args.input, args.output_npz, args.output_ply)
    input_file_path = r"F:\20240731_UAV_G3033\Aprojects\9\test\G3033_9_Part1_smallpart.ply"
    output_npz_path = input_file_path.replace('.ply', '_triangulation_data.npz')
    output_ply_path = input_file_path.replace('.ply', '_triangulation_mesh.ply')
    main(input_file_path, output_npz_path, output_ply_path)

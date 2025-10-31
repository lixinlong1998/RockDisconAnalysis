import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, ConvexHull
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import struct

# Parameters
ANGLE_THRESHOLD_A = np.deg2rad(10)  # threshold a in radians
ANGLE_THRESHOLD_B = np.deg2rad(30)  # threshold b in radians


# Utility functions


def export_segments_ply(segments, filename):
    """Export line segments with length attribute to a PLY file."""
    vertices = []
    lines = []
    lengths = []
    for idx, (p1, p2, d) in enumerate(segments):
        vertices.append(p1)
        vertices.append(p2)
        lines.append((2 * idx, 2 * idx + 1))
        lengths.append(d)
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element edge {len(lines)}\n')
        f.write('property int vertex1\n')
        f.write('property int vertex2\n')
        f.write('property float length\n')
        f.write('end_header\n')
        for v in vertices:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for (i1, i2), d in zip(lines, lengths):
            f.write(f'{i1} {i2} {d}\n')


def load_point_cloud(path):
    """Load point cloud with normals and color from PLY or LAS."""
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def compute_surface_triangles(pcd):
    """Compute Delaunay triangulation on XY projection."""
    pts = np.asarray(pcd.points)
    xy = pts[:, :2]
    tri = Delaunay(xy)
    return tri.simplices


def compute_triangle_normals(pts, triangles):
    """Compute normals for each triangle face."""
    v0 = pts[triangles[:, 1]] - pts[triangles[:, 0]]
    v1 = pts[triangles[:, 2]] - pts[triangles[:, 0]]
    normals = np.cross(v0, v1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)
    return normals


def build_adjacency(triangles):
    """Build adjacency list: triangles sharing an edge."""
    edge_map = defaultdict(list)
    for tid, tri in enumerate(triangles):
        edges = [tuple(sorted((tri[i], tri[j]))) for i, j in [(0, 1), (1, 2), (2, 0)]]
        for e in edges:
            edge_map[e].append(tid)
    adj = defaultdict(set)
    for tids in edge_map.values():
        if len(tids) > 1:
            for t1 in tids:
                for t2 in tids:
                    if t1 != t2:
                        adj[t1].add(t2)
    return adj


def region_growing(triangles, normals, adjacency, seed_idx):
    """Perform region growing from a seed triangle."""
    visited = set()
    cluster = []
    queue = deque([seed_idx])
    seed_normal = normals[seed_idx]
    sum_normal = seed_normal.copy()
    count = 1
    while queue:
        tid = queue.popleft()
        if tid in visited:
            continue
        visited.add(tid)
        nrm = normals[tid]
        if np.arccos(np.clip(np.dot(nrm, seed_normal), -1.0, 1.0)) > ANGLE_THRESHOLD_A:
            continue
        avg_nrm = sum_normal / count
        if np.arccos(np.clip(np.dot(nrm, avg_nrm), -1.0, 1.0)) > ANGLE_THRESHOLD_B:
            continue
        cluster.append(tid)
        sum_normal += nrm
        count += 1
        for nei in adjacency[tid]:
            if nei not in visited:
                queue.append(nei)
    return cluster


def fit_plane(pts):
    """Fit plane Ax+By+Cz+D=0 to points via SVD."""
    centroid = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[2, :]
    D = -normal.dot(centroid)
    return normal[0], normal[1], normal[2], D


def compute_orientation(A, B, C):
    """Compute dip direction and dip in degrees."""
    if A == 0 and B == 0:
        dip_dir = 0.0
    else:
        dip_dir = (np.degrees(np.arctan2(A, B)) + 360) % 360
    dip = 90 - np.degrees(np.arccos(abs(C)))
    return dip_dir, dip


def compute_trace(pts):
    """Compute the longest segment endpoints and its length from a point set."""
    hull = ConvexHull(pts[:, :2])
    hull_pts = pts[hull.vertices]
    diffs = hull_pts[:, None, :] - hull_pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return hull_pts[i], hull_pts[j], dists[i, j]


def write_mesh_binary_ply(filename, pts, triangles, face_attrs):
    """Write mesh with face attributes (label, dip_dir, dip) in binary PLY."""
    num_v = len(pts)
    num_f = len(triangles)
    header = [
        'ply',
        'format binary_little_endian 1.0',
        f'element vertex {num_v}',
        'property float x', 'property float y', 'property float z',
        f'element face {num_f}',
        'property list uchar int vertex_indices',
        'property int label',
        'property float dip_dir', 'property float dip',
        'end_header'
    ]
    with open(filename, 'wb') as f:
        f.write(("\n".join(header) + "\n").encode('utf-8'))
        # vertices
        for v in pts:
            f.write(struct.pack('<3f', *v))
        # faces
        for idx, tri in enumerate(triangles):
            lbl, dip_dir, dip = face_attrs[idx]
            f.write(struct.pack('<B3i', 3, int(tri[0]), int(tri[1]), int(tri[2])))
            f.write(struct.pack('<i2f', int(lbl), dip_dir, dip))


def main(input_path, output_prefix):
    # load point cloud
    pcd = load_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    print(f"[DEBUG] Loaded point cloud with {len(pts)} points")

    # triangulation
    triangles = compute_surface_triangles(pcd)
    print(f"[DEBUG] Generated {len(triangles)} triangles")

    normals = compute_triangle_normals(pts, triangles)
    print(f"[DEBUG] Computed normals array of shape {normals.shape}")

    adjacency = build_adjacency(triangles)
    print(f"[DEBUG] Built adjacency for {len(adjacency)} triangles")

    # cluster faces
    labels = -np.ones(len(triangles), dtype=int)
    curr_label = 0
    for tid in range(len(triangles)):
        if labels[tid] != -1:
            continue
        cluster = region_growing(triangles, normals, adjacency, seed_idx=tid)
        for t in cluster:
            labels[t] = curr_label
        print(f"[DEBUG] Cluster {curr_label} size: {len(cluster)}")
        curr_label += 1
    print(f"[DEBUG] Total clusters: {curr_label}")

    # parallel compute face attributes
    def face_attr(tid):
        tri_pts = pts[triangles[tid]]
        A, B, C, D = fit_plane(tri_pts)
        dip_dir, dip = compute_orientation(A, B, C)
        return tid, labels[tid], dip_dir, dip

    face_attrs = [None] * len(triangles)
    with ProcessPoolExecutor() as executor:
        for tid, lbl, dip_dir, dip in executor.map(face_attr, range(len(triangles))):
            face_attrs[tid] = (lbl, dip_dir, dip)
    print(f"[DEBUG] Computed face attributes for first 5 faces: {face_attrs[:5]}")

    # export mesh binary PLY with face attributes
    mesh_file = f"{output_prefix}_mesh.ply"
    write_mesh_binary_ply(mesh_file, pts, triangles, face_attrs)
    print(f"[DEBUG] Exported binary mesh PLY: {mesh_file}")

    # compute and export trace segments
    clusters = [np.unique(triangles[np.where(labels == i)[0]].ravel()) for i in range(curr_label)]

    def seg_item(idx_pts):
        subset_pts = pts[idx_pts]
        return compute_trace(subset_pts)

    segments = []
    with ProcessPoolExecutor() as executor:
        for (end1, end2, d), lbl in zip(executor.map(seg_item, clusters), range(curr_label)):
            segments.append((end1, end2, d))
            print(f"[DEBUG] Segment for cluster {lbl}: length={d}")

    # export segments as PLY
    seg_file = f"{output_prefix}_segments.ply"
    export_segments_ply(segments, seg_file)
    print(f"[DEBUG] Exported segments PLY: {seg_file}")


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True, help='Input point cloud (.ply/.las)')
    # parser.add_argument('--output_prefix', default='output', help='Prefix for output files')
    # args = parser.parse_args()
    # main(args.input, args.output_prefix)
    input_file_path = r"F:\20240731_UAV_G3033\Aprojects\9\G3033_9_Part1_smallpart.ply"
    output_file_path = input_file_path.replace('.ply', '')
    main(input_file_path, output_file_path)

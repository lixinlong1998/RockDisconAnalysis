# EdgeBoundaryDetect_PLY.py
# Python 3.9+ 兼容；依赖: numpy, scipy, plyfile
import os
import time
import argparse
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

'''
# 1) 默认参数，在输入同目录输出：
python _EdgeDetection.py `
--input "E:\Database\_RockPoints\TSDK_Rockfall_RegularClip\TSDK_Rockfall_13_P1_ORG.ply" `
--out_dir "D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\TSDK_Rockfall_13_P1_ORG_workspace"
'''


# ---------------- I/O: PLY ----------------
def read_ply_points_normals(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    读取 PLY 点云:
      返回 (points[N,3], normals[N,3] or None, vertex_struct_array[N])
    说明:
      - 尽量保留原顶点属性（vertex_struct_array），后续筛选子集写回能保留原字段
      - 如果 ply 里没有 (nx,ny,nz)，返回 normals=None
    """
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError("PLY 中没有 'vertex' 元素。")

    v = ply["vertex"].data  # structured array
    names = v.dtype.names
    if names is None or not set(["x", "y", "z"]).issubset(names):
        raise ValueError("vertex 中缺少 x/y/z 坐标。")

    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    if {"nx", "ny", "nz"}.issubset(names):
        nrm = np.vstack([v["nx"], v["ny"], v["nz"]]).T.astype(np.float64)
    else:
        nrm = None
    return pts, nrm, v


def write_ply_vertices(path: str, vertex_struct_array: np.ndarray):
    """
    将结构化的顶点数组写为 PLY（仅包含 vertex）。
    这里直接用原 dtype 写出，尽量保留原属性（包含颜色等）。
    """
    # 注意：subset 的 structured array 可能是视图，这里复制一份保证连续存储
    vertex_copy = np.array(vertex_struct_array, copy=True)
    el = PlyElement.describe(vertex_copy, "vertex")
    PlyData([el], text=False).write(path)


# ---------------- 法向估计与一致化 ----------------
def estimate_normals_pca(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    基于 kNN-PCA 的法向估计（未组织点云适用）。
    """
    n = points.shape[0]
    k = max(3, min(k, n))  # 防止 k > n
    tree = cKDTree(points)
    normals = np.zeros_like(points, dtype=np.float64)
    _, idxs = tree.query(points, k=k)
    for i in range(n):
        nbr = points[idxs[i]]
        X = nbr - nbr.mean(axis=0, keepdims=True)
        C = X.T @ X
        # 最小特征值对应特征向量为法向
        w, v = np.linalg.eigh(C)
        nvec = v[:, 0]
        nrm = np.linalg.norm(nvec)
        normals[i] = nvec / (nrm + 1e-12)
    return normals


def orient_normals_consistently(points: np.ndarray,
                                normals: np.ndarray,
                                k_graph: int = 10,
                                viewpoint: Optional[np.ndarray] = None) -> np.ndarray:
    """
    法向一致化：
    - 若给 viewpoint（相机/观察点），统一指向该视点；
    - 否则在 kNN 图上做 BFS，让相邻法向尽量同向。
    """
    nf = normals.copy()
    if viewpoint is not None:
        to_cam = (viewpoint.reshape(1, 3) - points)
        flip = (np.sum(nf * to_cam, axis=1) < 0.0)
        nf[flip] *= -1.0
        return nf

    n = points.shape[0]
    k_graph = max(2, min(k_graph, n))
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=k_graph)

    visited = np.zeros(n, dtype=bool)
    for s in range(n):  # 处理不连通
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        while stack:
            i = stack.pop()
            for j in idxs[i][1:]:
                if not visited[j]:
                    if np.dot(nf[i], nf[j]) < 0:
                        nf[j] *= -1.0
                    visited[j] = True
                    stack.append(j)
    return nf


# ---------------- 球形边界检测 ----------------
def spherical_boundary_detection(points: np.ndarray,
                                 normals: Optional[np.ndarray] = None,
                                 r_ball: float = 0.1,
                                 angle_thr_deg: float = 80.0,
                                 min_neighbors: int = 15,
                                 mark_all_neighbors: bool = True,
                                 sample_cap: int = 128,
                                 approx: bool = False,
                                 knn_normals: int = 30,
                                 viewpoint: Optional[np.ndarray] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    固定半径球形算子边界检测：
      - 对每个点取 r_ball 邻域，计算邻域法向的最大两两夹角；
      - 若 > angle_thr_deg，则记为“边界”。
    返回:
      boundary_mask: (N,) bool
      used_normals:  (N,3)
    """
    N = points.shape[0]
    if normals is None:
        normals = estimate_normals_pca(points, k=knn_normals)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    normals = orient_normals_consistently(points, normals, k_graph=10, viewpoint=viewpoint)

    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, r_ball)

    boundary = np.zeros(N, dtype=bool)
    thr_rad = np.deg2rad(angle_thr_deg)

    for i, idx in enumerate(neighborhoods):
        m = len(idx)
        if m < min_neighbors:
            continue

        # 采样限制，避免 O(m^2) 过大
        if m > sample_cap:
            sel = np.random.choice(m, size=sample_cap, replace=False)
            idx = [idx[s] for s in sel]
            m = len(idx)

        U = normals[idx]  # (m,3)

        if approx:
            # O(m) 近似：双重 farthest-on-sphere
            anchor = U[0]
            dots = U @ anchor
            a = U[np.argmin(dots)]
            dots2 = U @ a
            b = U[np.argmin(dots2)]
            min_dot = float(np.dot(a, b))
        else:
            # 精确：两两最小点积（最大夹角），O(m^2)
            M = U @ U.T
            min_dot = np.clip(M.min(), -1.0, 1.0)

        max_angle = np.arccos(np.clip(min_dot, -1.0, 1.0))
        if max_angle >= thr_rad:
            if mark_all_neighbors:
                boundary[idx] = True
            else:
                boundary[i] = True

    return boundary, normals


# ---------------- 主流程（CLI） ----------------
def main():
    parser = argparse.ArgumentParser(
        description="固定半径球形算子进行点云边界检测（从 PLY 读 / 分别写出边界点与非边界点）")
    parser.add_argument("--input", type=str, default=None, help="输入点云 .ply")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认同输入）")
    parser.add_argument("--prefix", type=str, default=None, help="输出文件名前缀（默认用输入文件名）")

    # 算法参数
    parser.add_argument("--radius", type=float, default=0.2, help="球半径 r_ball")
    parser.add_argument("--angle_deg", type=float, default=80.0, help="阈值：邻域最大法向夹角（度）")
    parser.add_argument("--min_neighbors", type=int, default=20, help="最小邻域点数")
    parser.add_argument("--mark_all_neighbors", action="store_true",
                        help="若触发阈值，将邻域内点全部标为边界（否则只标球心）")
    parser.add_argument("--approx", action="store_true",
                        help="启用 O(m) 近似（更快）")
    parser.add_argument("--sample_cap", type=int, default=128,
                        help="邻域采样上限（限制 O(m^2) 的 m）")
    parser.add_argument("--knn_normals", type=int, default=30, help="PCA 法向估计的 k（无法向时使用）")

    # 法向一致化
    parser.add_argument("--viewpoint", type=float, nargs=3, default=None,
                        help="可选：视点坐标 x y z，用于法向朝向一致化")
    args = parser.parse_args()

    in_path = args.input
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(in_path))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(in_path))[0]
    prefix = args.prefix or stem

    t0 = time.perf_counter()
    print(f"[I] 读取: {in_path}")
    points, normals, vertex_struct = read_ply_points_normals(in_path)
    print(f"[I] 点数: {points.shape[0]:,d} | 已有法向: {normals is not None}")

    vp = np.array(args.viewpoint, dtype=float) if args.viewpoint is not None else None

    # 检测
    t1 = time.perf_counter()
    boundary_mask, used_normals = spherical_boundary_detection(
        points, normals,
        r_ball=args.radius,
        angle_thr_deg=args.angle_deg,
        min_neighbors=args.min_neighbors,
        mark_all_neighbors=args.mark_all_neighbors,
        sample_cap=args.sample_cap,
        approx=args.approx,
        knn_normals=args.knn_normals,
        viewpoint=vp
    )
    t2 = time.perf_counter()
    ratio = boundary_mask.mean()
    print(
        f"[I] 检测完成: 边界点 {boundary_mask.sum():,d} / {len(boundary_mask):,d} ({ratio:.3%}) | 用时 {(t2 - t1):.2f}s")

    # 若原 PLY 无法向，可选在这里把估计/一致化后的法向写回（保持字段名 nx,ny,nz）
    v = vertex_struct
    names = v.dtype.names or ()
    need_add_normals = not {"nx", "ny", "nz"}.issubset(names)
    if need_add_normals:
        # 给两个输出都添加 nx,ny,nz 字段（为了可视化更方便）
        v = append_normals_to_struct(v, used_normals)
        print("[I] 原文件无法向字段，已把估计法向 (nx,ny,nz) 加入输出。")

    # 拆分并写出
    v_edge = v[boundary_mask]
    v_non = v[~boundary_mask]

    out_edge = os.path.join(out_dir, f"{prefix}_edge.ply")
    out_non = os.path.join(out_dir, f"{prefix}_nonedge.ply")

    write_ply_vertices(out_edge, v_edge)
    write_ply_vertices(out_non, v_non)

    t3 = time.perf_counter()
    print(f"[I] 已写出:\n  - {out_edge}  ({len(v_edge):,d} 点)\n  - {out_non}  ({len(v_non):,d} 点)")
    print(f"[I] 总用时: {(t3 - t0):.2f}s")


def append_normals_to_struct(vertex_struct: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    给现有的 structured array 添加 nx, ny, nz 三个字段；返回新的 structured array。
    - 若已有字段则直接覆盖；若无则扩展 dtype。
    """
    v = vertex_struct
    names = v.dtype.names or ()
    if {"nx", "ny", "nz"}.issubset(names):
        v = v.copy()
        v["nx"] = normals[:, 0]
        v["ny"] = normals[:, 1]
        v["nz"] = normals[:, 2]
        return v

    # 构造新的 dtype（在末尾追加三列 float32）
    old_descr = v.dtype.descr
    new_descr = old_descr + [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    v_new = np.empty(v.shape, dtype=new_descr)
    # 复制旧字段
    for name in v.dtype.names:
        v_new[name] = v[name]
    # 新字段赋值
    v_new["nx"] = normals[:, 0].astype(np.float32)
    v_new["ny"] = normals[:, 1].astype(np.float32)
    v_new["nz"] = normals[:, 2].astype(np.float32)
    return v_new


if __name__ == "__main__":
    main()

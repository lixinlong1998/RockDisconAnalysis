# -*- coding: utf-8 -*-
"""
BlockVerticesExport.py
目的：在保持 In-Situ 2021 方法 Step5~Step6 逻辑不变的前提下，只输出“块体顶点”到可视化目录。
要点：
 - 邻接判定：提供 'global'（全局 KDTree）与 'centers'（中心预筛 + 成对 query_ball_tree）两种快速方案
 - 顶点枚举：纯三平面求交（不使用椭圆盘裁剪）+ 全局外包体过滤（AABB/ConvexHull）
 - 去重：基于 merge_tol 的栅格量化 O(1) 去重
 - 导出：每个顶点单独一个 ASCII PLY，文件名为 [顶点-<A>,<B>,<C>].ply

依赖：numpy, scipy.spatial
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from scipy.spatial import cKDTree, ConvexHull


# --------------------------
# 数据结构
# --------------------------
@dataclass
class BlockVertex:
    coord: np.ndarray  # (3,)
    planes: Tuple[int, int, int]  # 三个结构面索引（升序）


# --------------------------
# 工具：读平面、点、外包体
# --------------------------
def _get_plane_normed(disc):
    """返回 (n_unit, d_unit) 使得 n·x + d = 0 且 ||n||=1"""
    p = np.array(getattr(disc, 'plane_paras', getattr(disc, 'plane_params')), dtype=float)
    n = p[:3];
    d = p[3]
    ln = np.linalg.norm(n) + 1e-12
    return n / ln, d / ln


def _collect_disc_points(disc) -> np.ndarray:
    if getattr(disc, 'rock_points', None) is None:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array([p.coord for p in disc.rock_points.points], dtype=np.float32)


def _gather_all_coords_from_discs(discs, max_sample_per_disc=2000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    for d in discs:
        P = _collect_disc_points(d)
        if P.shape[0] == 0:
            continue
        if P.shape[0] > max_sample_per_disc:
            idx = rng.choice(P.shape[0], size=max_sample_per_disc, replace=False)
            P = P[idx]
        pts.append(P)
    return np.vstack(pts) if len(pts) > 0 else np.zeros((0, 3), dtype=np.float32)


def prepare_envelope(discs, method='aabb', aabb_pad=0.05, hull_sample=40000, seed=0):
    """AABB（快）或 ConvexHull（更准）作为全局外包体，用于过滤岩体外部伪交点。"""
    A = _gather_all_coords_from_discs(discs, max_sample_per_disc=max(1, hull_sample // max(1, len(discs))), seed=seed)
    if A.shape[0] == 0:
        return {'type': 'aabb', 'min': np.array([-1e9] * 3), 'max': np.array([1e9] * 3), 'tol': 1e-6}
    if method == 'convex_hull' and A.shape[0] >= 16:
        try:
            hull = ConvexHull(A)
            return {'type': 'hull', 'H': hull.equations.copy(), 'tol': 1e-6}
        except Exception:
            pass
    m = np.min(A, axis=0) - aabb_pad
    M = np.max(A, axis=0) + aabb_pad
    return {'type': 'aabb', 'min': m, 'max': M, 'tol': 1e-6}


def point_in_envelope(P: np.ndarray, env: dict) -> bool:
    if env['type'] == 'aabb':
        m, M = env['min'], env['max']
        return bool(np.all(P >= m) and np.all(P <= M))
    else:
        H = env['H'];
        tol = env['tol']
        return bool(np.all(H[:, :3] @ P + H[:, 3] <= tol))


# --------------------------
# 邻接（Step5）两种快速方案
# --------------------------
def build_neighbors_global_kdtree(discs, interdist: float, max_pts_per_disc: int = 256, seed: int = 0):
    """一次全局 KDTree：抽样后半径搜索，得到跨面邻接。"""
    rng = np.random.default_rng(seed)
    pts = [];
    owners = []
    for di, d in enumerate(discs):
        P = _collect_disc_points(d)
        if P.shape[0] == 0:
            continue
        # if P.shape[0] > max_pts_per_disc:
        #     idx = rng.choice(P.shape[0], size=max_pts_per_disc, replace=False)
        #     P = P[idx]
        pts.append(P)
        owners.append(np.full(P.shape[0], di, dtype=np.int32))
    neighbors = {i: set() for i in range(len(discs))}
    if len(pts) == 0:
        return neighbors
    X = np.vstack(pts);
    O = np.concatenate(owners)
    tree = cKDTree(X)
    nbrs = tree.query_ball_point(X, r=interdist, workers=-1)
    for idx, lst in enumerate(nbrs):
        oi = O[idx]
        if not lst: continue
        for j in lst:
            if j == idx: continue
            oj = O[j]
            if oi != oj:
                neighbors[oi].add(int(oj));
                neighbors[oj].add(int(oi))
    return neighbors


def build_neighbors_centers_prefilter(discs, interdist: float, sample_per_disc: int = 512):
    """中心 KDTree 预筛 + 成对 query_ball_tree：更稳健的大数据方案。"""
    n = len(discs)
    centers = np.array([np.array(d.disc_center, dtype=np.float32) for d in discs])
    radii = np.array([max(float(getattr(d, 'ellip_a', 0.0)), float(getattr(d, 'ellip_b', 0.0))) for d in discs],
                     dtype=np.float32)
    Rmax = float(np.max(radii)) if n > 0 else 0.0
    ct = cKDTree(centers)
    candidates = {i: set() for i in range(n)}
    for i in range(n):
        rad = float(radii[i] + Rmax + interdist)
        idxs = ct.query_ball_point(centers[i], r=rad, workers=-1)
        for j in idxs:
            if j <= i: continue
            if np.linalg.norm(centers[i] - centers[j]) <= (radii[i] + radii[j] + interdist):
                candidates[i].add(j)
    kdts = []
    rng = np.random.default_rng(0)
    for d in discs:
        P = _collect_disc_points(d)
        if P.shape[0] == 0:
            kdts.append(None);
            continue
        if P.shape[0] > sample_per_disc:
            idx = rng.choice(P.shape[0], size=sample_per_disc, replace=False)
            P = P[idx]
        kdts.append(cKDTree(P))
    neighbors = {i: set() for i in range(n)}
    for i in range(n):
        kdt_i = kdts[i]
        if kdt_i is None: continue
        for j in candidates[i]:
            kdt_j = kdts[j]
            if kdt_j is None: continue
            hits = kdt_i.query_ball_tree(kdt_j, r=interdist)
            if any(len(lst) > 0 for lst in hits):
                neighbors[i].add(j);
                neighbors[j].add(i)
    return neighbors


# --------------------------
# 顶点（Step6）：三平面求交 + 外包过滤 + O(1) 去重
# --------------------------
def enumerate_vertices_plane_only(
        discs, neighbors: Dict[int, Set[int]],
        merge_tol: float,
        envelope: dict,
        angle_min_deg: float = 10.0,
        det_eps: float = 1e-10,
        use_solve: bool = True
) -> List[BlockVertex]:
    n = len(discs)
    # 预计算单位法向和平面常数
    N = np.zeros((n, 3), dtype=float)
    D = np.zeros((n,), dtype=float)
    for i, d in enumerate(discs):
        Ni, Di = _get_plane_normed(d)
        N[i] = Ni;
        D[i] = Di

    # 退化次序 + 前向邻接
    deg = np.array([len(neighbors[i]) for i in range(n)], dtype=int)
    order = np.argsort(deg, kind='mergesort')
    rank = np.empty(n, dtype=int);
    rank[order] = np.arange(n)
    fwd = [None] * n
    for i in range(n):
        fwd[i] = np.array(sorted([j for j in neighbors[i] if rank[j] > rank[i]]), dtype=int)

    # 栅格 O(1) 去重
    qinv = 1.0 / (merge_tol + 1e-12)
    bins = {}
    out: List[BlockVertex] = []
    sin_min = np.sin(np.deg2rad(angle_min_deg))

    def _try_add(P, trip):
        q = tuple(np.round(P * qinv).astype(np.int64))
        if q in bins: return
        bins[q] = 1
        out.append(BlockVertex(coord=P, planes=trip))

    for i in order:
        Ni, Di = N[i], D[i]
        Ni_fwd = fwd[i]
        if Ni_fwd.size < 2: continue
        for a in range(Ni_fwd.size):
            j = Ni_fwd[a]
            Nj, Dj = N[j], D[j]
            if np.linalg.norm(np.cross(Ni, Nj)) < sin_min:  # 角度筛#1
                continue
            Nj_fwd = fwd[j]
            W = np.intersect1d(Ni_fwd[a + 1:], Nj_fwd, assume_unique=True)
            if W.size == 0: continue
            for k in W:
                Nk, Dk = N[k], D[k]
                if (np.linalg.norm(np.cross(Ni, Nk)) < sin_min or
                        np.linalg.norm(np.cross(Nj, Nk)) < sin_min):  # 角度筛#2
                    continue
                # 三平面交点 A x = -d
                if use_solve:
                    A = np.stack([Ni, Nj, Nk], axis=0)
                    b = -np.array([Di, Dj, Dk], dtype=float)
                    try:
                        P = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        continue
                else:
                    c23 = np.cross(Nj, Nk)
                    denom = float(np.dot(Ni, c23))
                    if abs(denom) < det_eps:
                        continue
                    P = -(Di * c23 + Dj * np.cross(Nk, Ni) + Dk * np.cross(Ni, Nj)) / denom
                # 全局外包体过滤
                if not point_in_envelope(P, envelope):
                    continue
                _try_add(P.astype(np.float64), tuple(sorted((i, j, k))))
    return out


# --------------------------
# PLY 导出（单点）
# --------------------------
def _write_ply_point(path: str, point: np.ndarray, rgb: Tuple[int, int, int] = (255, 255, 255)):
    """保存单点 ASCII PLY（可视化兼容 CloudCompare / MeshLab）"""
    x, y, z = map(float, point)
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    header = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# --------------------------
# 主入口：只导出顶点到 PLY
# --------------------------
def export_vertices_insitu(
        discontinuitys,
        interdist: float,
        out_dir: str = "./visualization",
        avg_spacing: Optional[float] = None,
        neighbor_method: str = 'global',  # 'global' 或 'centers'
        max_pts_per_disc: int = 256,  # global 抽样
        sample_per_disc: int = 512,  # centers 抽样
        envelope_method: str = 'aabb',  # 'aabb' 或 'convex_hull'
        aabb_pad: float = 0.05,  # AABB 外扩
        hull_sample: int = 40000,  # hull 抽样
        vertex_merge_mult: float = 1.5,  # merge_tol = mult * (avg_spacing or interdist)
        angle_min_deg: float = 10.0,
        use_solve: bool = True,
        color_mode: str = "hash"  # 'hash' or 'white'
):
    """
    仅输出顶点 PLY（每个顶点一个文件：[顶点-A,B,C].ply）。
    返回：List[BlockVertex]
    """
    discs = discontinuitys.discontinuitys
    n = len(discs)
    if n == 0:
        os.makedirs(out_dir, exist_ok=True)
        return []

    # 合并公差
    merge_tol = vertex_merge_mult * float(avg_spacing if avg_spacing is not None else interdist)

    # 邻接
    neighbors = build_neighbors_global_kdtree(discs, interdist=interdist, max_pts_per_disc=max_pts_per_disc)

    # 外包体
    env = prepare_envelope(discs, method=envelope_method, aabb_pad=aabb_pad, hull_sample=hull_sample, seed=0)

    # 顶点
    verts = enumerate_vertices_plane_only(
        discs, neighbors,
        merge_tol=merge_tol,
        envelope=env,
        angle_min_deg=angle_min_deg,
        use_solve=use_solve
    )

    # 导出
    os.makedirs(out_dir, exist_ok=True)

    def _hash_color(a, b, c):
        if color_mode != "hash":
            return (255, 255, 255)
        seed = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
        rng = np.random.default_rng(seed)
        rgb = rng.integers(64, 255, size=3, dtype=np.int32)
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    for v in verts:
        A, B, C = v.planes
        name = f"顶点-{A},{B},{C}.ply"
        path = os.path.join(out_dir, name)
        _write_ply_point(path, v.coord, rgb=_hash_color(A, B, C))

    return verts

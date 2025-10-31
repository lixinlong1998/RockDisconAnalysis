# -*- coding: utf-8 -*-
"""
BlockInSitu2021.py  (Integrated, Fast, Plane-Only)
基于 2021 In-Situ Block Characterization 方法，从 Step5 开始实现：
 - Step5: 邻接判定（interdist）
 - Step6: 三平面交点（块体顶点）
 - Step7: 按共享面对聚合顶点 → 块体候选
 - Step8: 多面体建模（面内凸包 + 三角扇）

本版要点：
 - 不再使用椭圆盘裁剪顶点；三平面直接求交；
 - 使用全局外包体（AABB/ConvexHull）过滤岩体外部伪交点；
 - O(1) 栅格量化去重；
 - 两种高速邻接构图方案可选：'global' 或 'centers'。

依赖：
 - numpy, scipy.spatial (cKDTree, ConvexHull)

接口：
 - recognize_blocks_insitu(...)
   返回 List[BlockMesh]，并写回每个结构面的 d.block_id（int 或 None）
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from scipy.spatial import cKDTree, ConvexHull


# ======================
# 数据结构
# ======================
@dataclass
class BlockVertex:
    coord: np.ndarray  # (3,)
    planes: Tuple[int, int, int]  # 三个结构面索引（升序）


@dataclass
class BlockMesh:
    block_id: int
    face_ids: List[int]  # 本块体涉及的结构面索引集合（去重后排序）
    vertices: np.ndarray  # (V,3)
    faces: np.ndarray  # (F,3) 三角面片（顶点索引）


# ======================
# 工具：读取平面、点云、局部坐标系
# ======================
def _get_plane_normed(disc) -> Tuple[np.ndarray, float]:
    """读取平面参数并单位化法向；返回 (n, d)，满足 n·x + d = 0，且 ||n||=1"""
    p = np.array(getattr(disc, 'plane_paras', getattr(disc, 'plane_params')), dtype=float)
    n = p[:3];
    d = p[3]
    ln = np.linalg.norm(n) + 1e-12
    return n / ln, d / ln


def _disc_axes(disc, n_unit: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定结构面，返回其面内正交轴 (u,v)。优先使用已有属性（如 u_axis/long_axis_norm），否则从法向构造稳定基。
    """
    if n_unit is None:
        n_unit, _ = _get_plane_normed(disc)
    # 优先用已有轴
    if hasattr(disc, 'u_axis'):
        u = np.array(disc.u_axis, dtype=float)
    elif hasattr(disc, 'long_axis_norm'):
        u = np.array(disc.long_axis_norm, dtype=float)
    elif hasattr(disc, 'long_axis'):
        u = np.array(disc.long_axis, dtype=float)
    else:
        u = None
    if u is not None and np.linalg.norm(u) > 0:
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n_unit, u);
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            v = v / nv
            return u, v
    # 回退：任意稳定基
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_unit, ref);
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n_unit, u);
    v /= (np.linalg.norm(v) + 1e-12)
    return u, v


def _project_to_plane_local(p: np.ndarray, center: np.ndarray, n_unit: np.ndarray, u: np.ndarray, v: np.ndarray) -> \
Tuple[float, float]:
    """三维点 p 到以 center 为原点、(u,v) 为轴的局部平面坐标"""
    w = p - center
    return float(np.dot(w, u)), float(np.dot(w, v))


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


# ======================
# 外包体：AABB / Convex Hull
# ======================
def prepare_envelope(discs, method='aabb', aabb_pad=0.05, hull_sample=40000, seed=0):
    """
    准备全局外包体，用于过滤岩体外部伪交点
    method: 'aabb'（最快，默认）或 'convex_hull'（更准，稍慢）
    aabb_pad: AABB 外扩边距（米，建议≈ 2*点距）
    hull_sample: 构壳抽样点数上限
    """
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


# ======================
# 邻接构图（Step5）两种高速方案
# ======================
def build_neighbors_global_kdtree(discs, interdist: float, max_pts_per_disc: int = 256, seed: int = 0):
    """一次全局 KDTree：从每面抽样 <=S 点，建一棵树，半径搜索得到跨面邻接。最快。"""
    rng = np.random.default_rng(seed)
    pts = []
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
    X = np.vstack(pts)
    O = np.concatenate(owners)
    tree = cKDTree(X)
    nbrs = tree.query_ball_point(X, r=interdist, workers=-1)
    for idx, lst in enumerate(nbrs):
        oi = O[idx]
        if not lst:
            continue
        for j in lst:
            if j == idx:
                continue
            oj = O[j]
            if oi != oj:
                neighbors[oi].add(int(oj));
                neighbors[oj].add(int(oi))
    return neighbors


def build_neighbors_centers_prefilter(discs, interdist: float, sample_per_disc: int = 512):
    """
    中心 KDTree 预筛 + 成对 query_ball_tree：
      1) 用中心-半径粗筛候选面对
      2) 对候选面对构建 KDTree，使用 query_ball_tree(r=interdist) 判定邻接
    """
    n = len(discs)
    centers = np.array([np.array(d.disc_center, dtype=np.float32) for d in discs])
    radii = np.array([max(float(getattr(d, 'ellip_a', 0.0)), float(getattr(d, 'ellip_b', 0.0))) for d in discs],
                     dtype=np.float32)
    Rmax = float(np.max(radii)) if n > 0 else 0.0

    # 1) 中心预筛
    ct = cKDTree(centers)
    candidates = {i: set() for i in range(n)}
    for i in range(n):
        rad = float(radii[i] + Rmax + interdist)
        idxs = ct.query_ball_point(centers[i], r=rad, workers=-1)
        for j in idxs:
            if j <= i: continue
            if np.linalg.norm(centers[i] - centers[j]) <= (radii[i] + radii[j] + interdist):
                candidates[i].add(j)

    # 2) 面点 KDTree（抽样）
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
        if kdt_i is None:
            continue
        for j in candidates[i]:
            kdt_j = kdts[j]
            if kdt_j is None:
                continue
            hits = kdt_i.query_ball_tree(kdt_j, r=interdist)
            if any(len(lst) > 0 for lst in hits):
                neighbors[i].add(j);
                neighbors[j].add(i)
    return neighbors


# ======================
# 顶点枚举（Step6）：纯三平面交点 + 外包过滤 + O(1) 去重
# ======================
def _enumerate_block_vertices_plane_only_fast(
        discs, neighbors: Dict[int, Set[int]],
        merge_tol: float,
        envelope: dict,
        angle_min_deg: float = 8.0,
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

    # 退化次序（按度数升序）+ 前向邻接
    deg = np.array([len(neighbors[i]) for i in range(n)], dtype=int)
    order = np.argsort(deg, kind='mergesort')
    rank = np.empty(n, dtype=int);
    rank[order] = np.arange(n)
    fwd = [None] * n
    for i in range(n):
        fwd[i] = np.array(sorted([j for j in neighbors[i] if rank[j] > rank[i]]), dtype=int)

    # 栅格量化去重
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
        if Ni_fwd.size < 2:
            continue
        for a in range(Ni_fwd.size):
            j = Ni_fwd[a]
            Nj, Dj = N[j], D[j]
            # 角度快速筛 #1
            if np.linalg.norm(np.cross(Ni, Nj)) < sin_min:
                continue
            Nj_fwd = fwd[j]
            # 交集（仅 rank 更大的 k）
            W = np.intersect1d(Ni_fwd[a + 1:], Nj_fwd, assume_unique=True)
            if W.size == 0:
                continue
            for k in W:
                Nk, Dk = N[k], D[k]
                # 角度快速筛 #2
                if (np.linalg.norm(np.cross(Ni, Nk)) < sin_min or
                        np.linalg.norm(np.cross(Nj, Nk)) < sin_min):
                    continue

                # 三平面求交：A x = -d
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

                # 外包体过滤：去掉岩体外的伪交点
                if not point_in_envelope(P, envelope):
                    continue

                _try_add(P.astype(np.float64), tuple(sorted((i, j, k))))

    return out


# ======================
# 顶点聚合（Step7）
# ======================
def _build_vertex_graph(verts: List[BlockVertex]) -> Dict[int, Set[int]]:
    """
    若两个顶点共享“两个相同结构面”（位于同一条交线上），连一条边。
    """
    g = {i: set() for i in range(len(verts))}
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for vid, v in enumerate(verts):
        a, b, c = v.planes
        for pair in [(a, b), (a, c), (b, c)]:
            key = tuple(sorted(pair))
            buckets.setdefault(key, []).append(vid)
    # 在每条交线上将同线顶点两两连接（或可改为按空间近邻连链，减少团规模）
    for vids in buckets.values():
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                g[vids[i]].add(vids[j]);
                g[vids[j]].add(vids[i])
    return g


def _connected_components(g: Dict[int, Set[int]]) -> List[List[int]]:
    comps = []
    visited = set()
    for s in g.keys():
        if s in visited:
            continue
        stack = [s];
        visited.add(s);
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in g[u]:
                if v not in visited:
                    visited.add(v);
                    stack.append(v)
        comps.append(comp)
    return comps


# ======================
# 建模（Step8）：逐面 2D 凸包 + 三角扇
# ======================
def _triangulate_block(discs, verts: List[BlockVertex], comp: List[int]) -> Optional[BlockMesh]:
    """
    输入：一个连通分量 comp（顶点索引列表）
    输出：BlockMesh（若构面失败则返回 None）
    """
    if len(comp) < 4:
        return None

    # 顶点集
    P = np.vstack([verts[i].coord for i in comp]).astype(np.float32)  # (V,3)

    # 面 → 顶点列表
    plane_to_vertices: Dict[int, List[int]] = {}
    for local_idx, vid in enumerate(comp):
        a, b, c = verts[vid].planes
        for pid in [a, b, c]:
            plane_to_vertices.setdefault(pid, []).append(local_idx)

    vertices = P
    faces_tri = []
    face_ids_sorted = sorted(list(plane_to_vertices.keys()))
    for pid in face_ids_sorted:
        idxs = plane_to_vertices[pid]
        if len(idxs) < 3:
            continue
        disc = discs[pid]
        n_unit, _ = _get_plane_normed(disc)
        u, v = _disc_axes(disc, n_unit=n_unit)
        center = np.array(disc.disc_center, dtype=float)

        # 投到 2D
        pts2 = []
        for li in idxs:
            x, y = _project_to_plane_local(vertices[li], center, n_unit, u, v)
            pts2.append([x, y])
        pts2 = np.array(pts2, dtype=float)

        # 凸包索引
        try:
            ch = ConvexHull(pts2)
        except Exception:
            continue
        hull_order = ch.vertices.tolist()
        if len(hull_order) < 3:
            continue
        base = idxs[hull_order[0]]
        for k in range(1, len(hull_order) - 1):
            i1 = idxs[hull_order[k]]
            i2 = idxs[hull_order[k + 1]]
            faces_tri.append([base, i1, i2])

    if len(faces_tri) == 0:
        return None

    faces = np.array(faces_tri, dtype=np.int32)
    return BlockMesh(block_id=-1, face_ids=face_ids_sorted, vertices=vertices, faces=faces)


# ======================
# 主入口
# ======================
def recognize_blocks_insitu(
        discontinuitys,
        interdist: float,
        avg_spacing: Optional[float] = None,
        neighbor_method: str = 'global',  # 'global' 或 'centers'
        max_pts_per_disc: int = 256,  # 全局 KDTree 抽样上限
        sample_per_disc: int = 512,  # centers 方案每面抽样上限
        envelope_method: str = 'aabb',  # 'aabb' 或 'convex_hull'
        aabb_pad: float = 0.05,  # AABB 外扩（米）
        hull_sample: int = 40000,  # Hull 抽样总上限
        vertex_merge_mult: float = 1.5,  # 顶点合并公差 = mult * (avg_spacing or interdist)
        angle_min_deg: float = 10.0,  # 三面夹角快速筛阈值
        min_vertices_per_block: int = 6,  # 最小顶点数过滤
        use_solve: bool = True  # True: np.linalg.solve；False: 解析式
) -> List[BlockMesh]:
    """
    识别块体（Step5~Step8，纯三平面求交版本）
    """
    discs = discontinuitys.discontinuitys
    n = len(discs)
    if n == 0:
        return []

    # 合并公差
    merge_tol = vertex_merge_mult * float(avg_spacing if avg_spacing is not None else interdist)

    # 邻接（Step5）
    if neighbor_method == 'centers':
        neighbors = build_neighbors_centers_prefilter(discs, interdist=interdist, sample_per_disc=sample_per_disc)
    else:
        neighbors = build_neighbors_global_kdtree(discs, interdist=interdist, max_pts_per_disc=max_pts_per_disc)

    # 外包体（一次构建，全流程复用）
    env = prepare_envelope(discs, method=envelope_method, aabb_pad=aabb_pad, hull_sample=hull_sample, seed=0)

    # 顶点（Step6）
    verts = _enumerate_block_vertices_plane_only_fast(
        discs, neighbors,
        merge_tol=merge_tol,
        envelope=env,
        angle_min_deg=angle_min_deg,
        use_solve=use_solve
    )

    if len(verts) == 0:
        return []

    # 顶点聚合（Step7）
    vg = _build_vertex_graph(verts)
    comps = _connected_components(vg)

    # 建模（Step8）
    blocks: List[BlockMesh] = []
    bid = 0
    for comp in comps:
        if len(comp) < max(4, min_vertices_per_block):
            continue
        bm = _triangulate_block(discs, verts, comp)
        if bm is None:
            continue
        bm.block_id = bid
        blocks.append(bm)
        # 回填 block_id（供分析/导出）
        for fid in bm.face_ids:
            setattr(discs[fid], 'block_id', bid)
        bid += 1

    return blocks

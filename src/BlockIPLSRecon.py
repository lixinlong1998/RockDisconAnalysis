# -*- coding: utf-8 -*-
"""
BlockIPLSRecon.py
实现：基于“平面半空间融合”（half-space fusion）的自动块体识别
依赖你现有的数据结构：
 - Discontinuity.Discontinuity（含：disc_center, normal, plane_params(A,B,C,D), ellip_a, ellip_b, polygon_area, rock_points等）
 - PointCloud.RockPointCloud / Point3D（点的 coord, plane_params 等）
输出：
 - blocks: List[BlockResult]（含 vertices, faces），可直接 Export.export_all_blocks()
 - 同时在每个 discontinuity 上写入：disc.type ∈ {'freeface','jointface','undefined'} 和 disc.block_id ∈ {0,1,2,...或None}
"""
from typing import Set
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# -----------------------------
# 数据结构：块体网格结果
# -----------------------------
@dataclass
class BlockResult:
    block_id: int
    face_ids: List[int]  # 参与构壳的结构面在 discontinuitys 列表中的索引
    vertices: np.ndarray  # (V,3)
    faces: np.ndarray  # (F,3)  以三角面片三元组（顶点索引）表示


# -----------------------------
# 工具函数
# -----------------------------


def _bron_kerbosch_pivot(R: Set[int], P: Set[int], X: Set[int], adj: Dict[int, Set[int]], res: List[Set[int]]):
    """Bron–Kerbosch with pivoting to find maximal cliques."""
    if not P and not X:
        res.append(set(R))
        return
    # 选择支点以加速
    u = next(iter(P | X)) if (P or X) else None
    Nu = adj[u] if u is not None else set()
    for v in list(P - Nu):
        _bron_kerbosch_pivot(R | {v}, P & adj[v], X & adj[v], adj, res)
        P.remove(v)
        X.add(v)

def _find_maximal_cliques(mutual_graph: Dict[int, Set[int]], min_size: int = 3) -> List[Set[int]]:
    """返回大小>=min_size 的最大团列表。"""
    # 转换邻接表为集合形式
    adj = {u: set(vs) for u, vs in mutual_graph.items()}
    V = set(adj.keys())
    res: List[Set[int]] = []
    _bron_kerbosch_pivot(set(), set(V), set(), adj, res)
    # 过滤小团
    res = [c for c in res if len(c) >= min_size]
    # 按大小降序
    res.sort(key=lambda s: (-len(s), sorted(list(s))))
    return res

def _plane_signed_distance(pts: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """ 计算点到平面的有符号距离；plane=[A,B,C,D]，法向为 (A,B,C)，距离符号按 Ax+By+Cz+D """
    A, B, C, D = plane
    denom = np.sqrt(A * A + B * B + C * C) + 1e-12
    return (pts @ np.array([A, B, C]) + D) / denom


def _estimate_avg_spacing_from_sample(all_coords: np.ndarray, k: int = 1, sample: int = 20000) -> float:
    """ 估计平均点距 d：从所有点中抽样，计算最近邻距离的中位数 """
    if all_coords.shape[0] > sample:
        idx = np.random.choice(all_coords.shape[0], size=sample, replace=False)
        coords = all_coords[idx]
    else:
        coords = all_coords
    kdt = cKDTree(coords)
    # k=2 最近邻，含自身；取第二个邻居的距离
    dists, _ = kdt.query(coords, k=2)
    nn = dists[:, 1]
    return float(np.median(nn))


def _build_neighbor_graph(discs, neighbor_threshold: float) -> Dict[int, List[int]]:
    """ 使用“球心距离 < 阈值”的邻接标准建立邻接表 """
    centers = np.array([d.disc_center for d in discs])
    kdt = cKDTree(centers)
    neighbors = {i: [] for i in range(len(discs))}
    for i, c in enumerate(centers):
        idxs = kdt.query_ball_point(c, r=neighbor_threshold)
        idxs.remove(i) if i in idxs else None
        neighbors[i] = sorted(idxs)
    return neighbors


def _unify_normals_via_mst(discs, neighbors: Dict[int, List[int]]):
    """
    基于邻接图构造最小生成树（MST），沿树传播翻转，使相邻法向尽量一致（dot>0）。
    注：这一步保证一致性，但“朝向岩体外部”的绝对意义需要根据你已有外部/内部判断再整体翻转一次（可选）。
    """
    n = len(discs)
    normals = np.array([d.normal for d in discs])  # (N,3)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    # 构造稀疏加权图：权重 = 1 - |dot(n_i, n_j)|
    rows, cols, data = [], [], []
    for i in range(n):
        for j in neighbors[i]:
            if j < n and j != i:
                w = 1.0 - abs(float(np.dot(normals[i], normals[j])))
                rows.append(i);
                cols.append(j);
                data.append(w)
                rows.append(j);
                cols.append(i);
                data.append(w)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    if graph.nnz == 0:
        return  # 孤立节点场景

    # MST（无向）——从面积最大的结构面作为根
    areas = np.array([getattr(d, 'polygon_area', 0.0) for d in discs])
    root = int(np.argmax(areas))
    mst = minimum_spanning_tree(graph)  # 返回有向稀疏矩阵（下三角或上三角）
    # 转成无向邻接
    coo = mst.tocoo()
    adj = {i: [] for i in range(n)}
    for u, v in zip(coo.row, coo.col):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    # BFS 传播翻转
    visited = [False] * n
    visited[root] = True
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if visited[v]:
                continue
            # 若 dot<0，则翻转 v 的法向与平面参数
            if float(np.dot(discs[u].normal, discs[v].normal)) < 0.0:
                discs[v].normal = -discs[v].normal
                A, B, C, D = discs[v].plane_params
                discs[v].plane_params = np.array([-A, -B, -C, -D], dtype=float)
            visited[v] = True
            queue.append(v)


def _gather_all_coords_from_discs(discs) -> np.ndarray:
    """ 将所有结构面的岩体点云合并为单一 ndarray (M,3) """
    coords = []
    for d in discs:
        if getattr(d, 'rock_points', None) is None:
            continue
        for p in d.rock_points.points:
            coords.append(p.coord)
    return np.array(coords, dtype=float) if coords else np.zeros((0, 3), dtype=float)


def _points_in_antihalf_and_close_to_plane(points: np.ndarray, plane: np.ndarray, normal: np.ndarray, tol: float):
    """
    选取位于“反法向半空间”(signed<0) 且 |距离|<tol 的点；返回掩码与距离
    """
    sd = _plane_signed_distance(points, plane)  # 有符号距离
    # 法向与 Ax+By+Cz+D 的符号方向是一致的（统一法向后）
    mask = (sd < 0) & (np.abs(sd) < tol)
    return mask, sd


def _collect_disc_points(disc) -> np.ndarray:
    """ 收集某结构面的全部点坐标 """
    if getattr(disc, 'rock_points', None) is None:
        return np.zeros((0, 3), dtype=float)
    return np.array([p.coord for p in disc.rock_points.points], dtype=float)


# -----------------------------
# 主流程
# -----------------------------
def recognize_blocks(discontinuitys,
                     neighbor_threshold: Optional[float] = None,
                     avg_spacing: Optional[float] = None,
                     include_frac: float = 0.20,
                     close_tol_mult: float = 2.0,
                     min_freefaces: int = 2,
                     build_mesh_method: str = 'convex_hull',
                     clique_shell: bool = True,
                     min_clique_size: int = 3,
                     exclusive_assignment: bool = True):
    """
    基于“平面半空间融合”的块体识别。
    关键新增：clique_shell=True 时，外壳按“互为 INBR 的完全子图（最大团）”生成。

    参数
    ----
    neighbor_threshold : 邻接阈值（球心距离）。None 则自动估计。
    avg_spacing : 平均点距 d。None 则自动估计（全局最近邻中位数）。
    include_frac : A 与 B 互为“包含”的比例阈值（工程量化）。
    close_tol_mult : 与平面距离阈值的倍率（默认 2d）。
    min_freefaces : 非 clique 模式下，判定 freeface 的最少互含邻面数（默认 2）。
    build_mesh_method : 'convex_hull'（默认），可自行扩展 alpha-shape 等。
    clique_shell : True 时启用“所有成员互为 INBR”的严格外壳规则。
    min_clique_size : clique 模式下的最小团大小（默认 3）。
    exclusive_assignment : True 则一个结构面只归属一个块体（按 clique 大小优先）。

    返回
    ----
    blocks : List[BlockResult]
    同时写回：
        disc.type ∈ {'freeface','jointface','undefined'}
        disc.block_id ∈ {0,1,2,..., None}
    """

    # ==== 内部：最大团（Bron–Kerbosch with pivoting） ====
    def _bron_kerbosch_pivot(R: Set[int], P: Set[int], X: Set[int],
                             adj: Dict[int, Set[int]], res: List[Set[int]]):
        if not P and not X:
            res.append(set(R))
            return
        u = next(iter(P | X)) if (P or X) else None
        Nu = adj[u] if u is not None else set()
        for v in list(P - Nu):
            _bron_kerbosch_pivot(R | {v}, P & adj[v], X & adj[v], adj, res)
            P.remove(v)
            X.add(v)

    def _find_maximal_cliques(mutual_graph: Dict[int, Set[int]], min_size: int = 3) -> List[Set[int]]:
        adj = {u: set(vs) for u, vs in mutual_graph.items()}
        V = set(adj.keys())
        res: List[Set[int]] = []
        _bron_kerbosch_pivot(set(), set(V), set(), adj, res)
        res = [c for c in res if len(c) >= min_size]
        res.sort(key=lambda s: (-len(s), sorted(list(s))))
        return res

    discs = discontinuitys.discontinuitys
    n = len(discs)
    if n == 0:
        return []

    # 估计平均点距 d
    if avg_spacing is None:
        all_coords = _gather_all_coords_from_discs(discs)
        if all_coords.shape[0] == 0:
            raise ValueError("没有可用点云来估计平均点距；请显式传入 avg_spacing。")
        avg_spacing = _estimate_avg_spacing_from_sample(all_coords, k=1, sample=20000)
    tol = close_tol_mult * avg_spacing

    # 邻接阈值自动估计（若未给）
    if neighbor_threshold is None:
        radii = np.array([max(getattr(d, 'ellip_a', 0.0), getattr(d, 'ellip_b', 0.0)) for d in discs])
        neighbor_threshold = 3.0 * float(np.median(radii) if np.any(radii > 0) else tol)

    # step1: 邻居（球心距离）
    neighbors = _build_neighbor_graph(discs, neighbor_threshold)

    # step2: MST 统一相邻法向方向（dot>0）
    _unify_normals_via_mst(discs, neighbors)

    # 预取面点与平面
    disc_points = [_collect_disc_points(d) for d in discs]
    disc_planes = [np.array(d.plane_params, dtype=float) for d in discs]
    disc_normals = [np.array(d.normal, dtype=float) for d in discs]

    # step3: 计算 INBRs（反法向 & |dist|<2d）
    INBRs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for A in range(n):
        A_plane, A_norm = disc_planes[A], disc_normals[A]
        for i in neighbors[A]:
            P = disc_points[i]
            if P.shape[0] == 0:
                continue
            mask, _ = _points_in_antihalf_and_close_to_plane(P, A_plane, A_norm, tol)
            if np.any(mask):
                INBRs[A].append(i)

    # step4: 互为包含（A in B 且 B in A），构 mutual_graph
    mutual_graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for A in range(n):
        A_pts = disc_points[A]
        if A_pts.shape[0] == 0:
            continue
        for B in INBRs[A]:
            # A⊂B ?
            B_plane, B_norm = disc_planes[B], disc_normals[B]
            mask_AinB, _ = _points_in_antihalf_and_close_to_plane(A_pts, B_plane, B_norm, tol)
            frac_AinB = (np.count_nonzero(mask_AinB) / max(1, A_pts.shape[0]))
            if frac_AinB < include_frac:
                continue
            # B⊂A ?
            B_pts = disc_points[B]
            if B_pts.shape[0] == 0:
                continue
            mask_BinA, _ = _points_in_antihalf_and_close_to_plane(B_pts, disc_planes[A], disc_normals[A], tol)
            frac_BinA = (np.count_nonzero(mask_BinA) / max(1, B_pts.shape[0]))
            if frac_BinA >= include_frac:
                mutual_graph[A].add(B)
                mutual_graph[B].add(A)

    # 初始化标注
    for d in discs:
        d.type = 'undefined'
        d.block_id = None

    # ===== 严格外壳：所有成员两两互为 INBR（完全子图/最大团） =====
    if clique_shell:
        cliques = _find_maximal_cliques(mutual_graph, min_size=min_clique_size)

        used: Set[int] = set()
        blocks: List[BlockResult] = []
        current_block_id = 0

        for C in cliques:
            if exclusive_assignment and (C & used):
                continue

            face_ids = sorted(list(C))
            # 标注 freeface 与 block_id
            for idx in face_ids:
                discs[idx].type = 'freeface'
                discs[idx].block_id = current_block_id

            # 可选：将与 clique 互含但不在 clique 内的面标记为 jointface（不归入该块）
            nbr_out = set()
            for u in C:
                nbr_out |= (mutual_graph[u] - C)
            for j in nbr_out:
                if getattr(discs[j], 'type', 'undefined') == 'undefined':
                    discs[j].type = 'jointface'

            # 组装壳点云：仅由 clique 内面自身点组成（严格定义下更干净）
            shell_pts = []
            for fidx in face_ids:
                P = disc_points[fidx]
                if P.shape[0] > 0:
                    shell_pts.append(P)
            if len(shell_pts) == 0:
                continue
            shell_pts = np.vstack(shell_pts)

            # 网格化
            try:
                if build_mesh_method == 'convex_hull':
                    hull = ConvexHull(shell_pts)
                    vertices = shell_pts.copy()
                    faces = hull.simplices.astype(np.int32)
                else:
                    # 需要的话在此扩展 alpha-shape 等
                    hull = ConvexHull(shell_pts)
                    vertices = shell_pts.copy()
                    faces = hull.simplices.astype(np.int32)
            except Exception:
                continue

            blocks.append(BlockResult(
                block_id=current_block_id,
                face_ids=face_ids,
                vertices=vertices.astype(np.float32),
                faces=faces.astype(np.int32)
            ))
            used |= C
            current_block_id += 1

        return blocks

    # ===== 非严格外壳：沿用“互含图的连通分量 + free/jointface”流程 =====
    disc_type = ['undefined'] * n
    for i in range(n):
        if len(mutual_graph[i]) >= min_freefaces:
            disc_type[i] = 'freeface'
        elif len(mutual_graph[i]) > 0:
            disc_type[i] = 'jointface'
        else:
            disc_type[i] = 'undefined'

    for i, d in enumerate(discs):
        d.type = disc_type[i]
        d.block_id = None

    visited = [False] * n
    blocks: List[BlockResult] = []
    current_block_id = 0

    # 从面积大到小构块
    for seed in np.argsort([-getattr(d, 'polygon_area', 0.0) for d in discs]):
        i = int(seed)
        if visited[i] or disc_type[i] == 'undefined':
            continue

        # 在互含图上 BFS 得到分量
        comp: Set[int] = set()
        queue = [i]
        visited[i] = True
        while queue:
            u = queue.pop(0)
            comp.add(u)
            for v in mutual_graph[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        if not any(disc_type[idx] == 'freeface' for idx in comp):
            continue

        comp_list = sorted(list(comp))
        freefaces = [idx for idx in comp_list if disc_type[idx] == 'freeface']
        jointfaces = [idx for idx in comp_list if disc_type[idx] == 'jointface']

        # 收集外壳点（以 freeface 为核心）
        shell_pts = []
        for fidx in freefaces:
            plane_f, norm_f = disc_planes[fidx], disc_normals[fidx]
            cand_idxs = set(neighbors[fidx]) | {fidx}
            for nb in cand_idxs:
                P = disc_points[nb]
                if P.shape[0] == 0:
                    continue
                mask, _ = _points_in_antihalf_and_close_to_plane(P, plane_f, norm_f, tol)
                if np.any(mask):
                    shell_pts.append(P[mask])

        if len(shell_pts) == 0:
            continue
        shell_pts = np.vstack(shell_pts)

        # 简化的“边缘点 + jointface 补壳”
        if len(jointfaces) > 0:
            jf_planes = [disc_planes[j] for j in jointfaces]
            jf_normals = [disc_normals[j] for j in jointfaces]
            jf_count = np.zeros(shell_pts.shape[0], dtype=int)
            for jp, jn in zip(jf_planes, jf_normals):
                sd = _plane_signed_distance(shell_pts, jp)
                jf_count += (np.abs(sd) < tol).astype(int)
            edge_mask = jf_count >= 1
            edge_pts = shell_pts[edge_mask]
            if edge_pts.shape[0] > 0:
                for j in jointfaces:
                    Pj = disc_points[j]
                    if Pj.shape[0] == 0:
                        continue
                    kdt = cKDTree(Pj)
                    idxs = kdt.query_ball_point(edge_pts, r=tol)
                    picked = np.unique([ii for lst in idxs for ii in lst])
                    if picked.size > 0:
                        shell_pts = np.vstack([shell_pts, Pj[picked]])

        # 网格化
        try:
            hull = ConvexHull(shell_pts)
            vertices = shell_pts.copy()
            faces = hull.simplices.astype(np.int32)
        except Exception:
            continue

        for idx in comp_list:
            discs[idx].block_id = current_block_id
        blocks.append(BlockResult(
            block_id=current_block_id,
            face_ids=comp_list,
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32)
        ))
        current_block_id += 1

    return blocks

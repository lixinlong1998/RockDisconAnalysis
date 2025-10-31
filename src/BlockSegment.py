# -*- coding: utf-8 -*-
"""
上述方法得到的岩块不太合理,可能的原因是没有充分利用结构面的圆盘模型,接下来请你阅读文献"2021-圆盘切割岩块-In-Situ Block Characterization of Jointed Rock Exposures Based on a 3D Point Cloud Model.pdf",然后我们重新写一个代码,这个代码在我的代码基础上实现该文章所提到的块体识别方法.

以下是我对In-Situ Block Characterization of Jointed Rock Exposures Based on a 3D Point Cloud Model这篇文章的理解,供你参考(不一定对,可以讨论):
Step 1 	使用CNN:Nesti-Net来估计点云的法向,文中宣称这种方法增加了鲁棒性和准确性
Step 2 	Normal转成Orientaion(dip direction 𝜃 and dip angle 𝛿)
Step 3 	使用fuzzy k-means算法(每个点到每个类都有一个隶属度,形成隶属度向量)对orientation聚类,而超过最大角度的点会被弃用
Step 4 	使用DBSCAN将子类划分出来
Step 5 	设置一个interdist参数,如果存在分别来自于不同discontinuity的2个点之间的距离小于interdist,则认为这两个discontinuity是邻接相交的.interdist的值稍比点云分辨率略高
Step 6 	寻找block vertex, 如果至少有3个互相相交的discontinuity, 则可以形成一个block vertex(A,B,C)
Step 7 	将属于一个block的block vertexs聚合到一起,在某些情况下,岩块可能被随机discontinuity部分切割,这样的discontinuity也可以聚合到一起
Step 8 	PCM-DDN: Polyhedral Modeling, 对于一些特殊情况，需要根据现场实际情况和地质人员的地质专业知识，利用人工平面来创建块体;
Step 9 	 in-situ block size distribution (IBSD)可以基于块体的四面体剖分方法进一步计算
在这些步骤中,step1-4在我们的原始代码中已经完成,我们只需要从Step5开始,给discontinuity寻找neighbors,然后寻找潜在的block vertex(A,B,C),聚合这些block vertex,然后用我们建立的discontinuity的elliptical disk(extention可以更大一些)来计算具体的vertex坐标,最后得到块体
"""

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
    face_ids: List[int]         # 参与构壳的结构面在 discontinuitys 列表中的索引
    vertices: np.ndarray        # (V,3)
    faces: np.ndarray           # (F,3)  以三角面片三元组（顶点索引）表示

# -----------------------------
# 工具函数
# -----------------------------
def _plane_signed_distance(pts: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """ 计算点到平面的有符号距离；plane=[A,B,C,D]，法向为 (A,B,C)，距离符号按 Ax+By+Cz+D """
    A, B, C, D = plane
    denom = np.sqrt(A*A + B*B + C*C) + 1e-12
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
                rows.append(i); cols.append(j); data.append(w)
                rows.append(j); cols.append(i); data.append(w)
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
    return np.array(coords, dtype=float) if coords else np.zeros((0,3), dtype=float)

def _points_in_antihalf_and_close_to_plane(points: np.ndarray, plane: np.ndarray, normal: np.ndarray, tol: float):
    """
    选取位于“反法向半空间”(signed<0) 且 |距离|<tol 的点；返回掩码与距离
    """
    sd = _plane_signed_distance(points, plane)   # 有符号距离
    # 法向与 Ax+By+Cz+D 的符号方向是一致的（统一法向后）
    mask = (sd < 0) & (np.abs(sd) < tol)
    return mask, sd

def _collect_disc_points(disc) -> np.ndarray:
    """ 收集某结构面的全部点坐标 """
    if getattr(disc, 'rock_points', None) is None:
        return np.zeros((0,3), dtype=float)
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
                     build_mesh_method: str = 'convex_hull'):
    """
    参数
    ----
    discontinuitys : Discontinuitys
        你的集合对象，含 .discontinuitys 列表（每个为 Discontinuity）
    neighbor_threshold : float
        邻接阈值，基于球心距离（单位与坐标一致）。若为 None，将用椭圆长短轴估计（见下）
    avg_spacing : float
        平均点距 d；若为 None 将自动估计（全体点最近邻中位数）
    include_frac : float
        “互为包含”判定时，A 的点有至少 include_frac 比例落入 B 的反法向且 |dist|<2d 视为 B 包含 A（反之同理）
    close_tol_mult : float
        “距离阈值”的倍率（默认 2d）
    min_freefaces : int
        至少满足互为包含的 INBR 数，判定为 freeface 的阈值（默认 2）
    build_mesh_method : str
        块体网格构建方式：'convex_hull'（默认）。你也可以后续扩展 'alpha_shape' 等。

    返回
    ----
    blocks : List[BlockResult]
    同时写回字段：
        disc.type ∈ {'freeface','jointface','undefined'}
        disc.block_id ∈ {0,1,2,..., None}
    """

    discs = discontinuitys.discontinuitys
    n = len(discs)
    if n == 0:
        return []

    # 估计 avg_spacing d
    if avg_spacing is None:
        all_coords = _gather_all_coords_from_discs(discs)
        if all_coords.shape[0] == 0:
            raise ValueError("没有可用点云来估计平均点距；请显式传入 avg_spacing。")
        avg_spacing = _estimate_avg_spacing_from_sample(all_coords, k=1, sample=20000)
    tol = close_tol_mult * avg_spacing  # 文中采用 2d，这里可调

    # 邻接阈值：若未给定，按椭圆半径经验估计（中心距 < 2*max(ellip_a,ellip_b) 的 2~3 倍）
    if neighbor_threshold is None:
        radii = np.array([max(getattr(d, 'ellip_a', 0.0), getattr(d, 'ellip_b', 0.0)) for d in discs])
        neighbor_threshold = 3.0 * float(np.median(radii) if np.all(radii>0) else tol)

    # step1: 中心球邻居
    neighbors = _build_neighbor_graph(discs, neighbor_threshold)

    # step2: 基于 MST 统一法向方向（使相邻 dot>0）
    _unify_normals_via_mst(discs, neighbors)

    # 预取每个面自身点 & plane
    disc_points = [ _collect_disc_points(d) for d in discs ]
    disc_planes = [ np.array(d.plane_params, dtype=float) for d in discs ]
    disc_normals = [ np.array(d.normal, dtype=float) for d in discs ]

    # step3: 计算 INBRs：对每个面A，遍历邻居 i 的点；在 A 的反法向半空间且 |dist|<2d 的点归入 A.included
    INBRs: Dict[int, List[int]] = {i: [] for i in range(n)}
    included_points_idx: Dict[Tuple[int,int], np.ndarray] = {}  # (A,i)-> mask idx（在 i 的点集中）
    for A in range(n):
        A_plane = disc_planes[A]
        A_norm = disc_normals[A]
        for i in neighbors[A]:
            if disc_points[i].shape[0] == 0:
                continue
            mask, _ = _points_in_antihalf_and_close_to_plane(disc_points[i], A_plane, A_norm, tol)
            if np.count_nonzero(mask) > 0:
                INBRs[A].append(i)
                included_points_idx[(A,i)] = np.where(mask)[0]

    # step4: 互为包含判断 + freeface / jointface 标注
    disc_type = ['undefined'] * n
    mutual_graph: Dict[int, Set[int]] = {i: set() for i in range(n)}  # 互为包含连边
    for A in range(n):
        A_pts = disc_points[A]
        if A_pts.shape[0] == 0:
            continue
        for B in INBRs[A]:
            # 判定：B 的反法向是否“包含” A 的点（部分或全部）
            # 即：A_pts 中落入 B 反法向 & |dist|<2d 的比例 >= include_frac
            B_plane, B_norm = disc_planes[B], disc_normals[B]
            mask_AinB, _ = _points_in_antihalf_and_close_to_plane(A_pts, B_plane, B_norm, tol)
            frac_AinB = (np.count_nonzero(mask_AinB) / max(1, A_pts.shape[0]))
            if frac_AinB >= include_frac:
                # 同时 B 也应在 A 的 INBR 列表中且满足比例
                B_pts = disc_points[B]
                mask_BinA, _ = _points_in_antihalf_and_close_to_plane(B_pts, disc_planes[A], disc_normals[A], tol)
                frac_BinA = (np.count_nonzero(mask_BinA) / max(1, B_pts.shape[0]))
                if frac_BinA >= include_frac:
                    mutual_graph[A].add(B)
                    mutual_graph[B].add(A)

    # 判定 freeface：若 A 至少有 min_freefaces 个“互为包含”的 INBR，则这些互为包含邻居都是 freeface；剩余邻居视为 jointface
    # 注意：这里 freeface/jointface 是“对 A 所属块”而言；简单处理：满足条件的各参与者都标 freeface
    # 后续我们再按“互为包含连通分量”聚合为块体
    for i in range(n):
        if len(mutual_graph[i]) >= min_freefaces:
            disc_type[i] = 'freeface'
        elif len(mutual_graph[i]) > 0:
            disc_type[i] = 'jointface'
        else:
            disc_type[i] = 'undefined'

    # 将类型写回
    for i, d in enumerate(discs):
        d.type = disc_type[i]   # Export.export_discon_analysis 里会读取该字段
        d.block_id = None       # 先占位，待真正成块后再回填

    # 基于“互为包含”图的连通分量，构造候选壳（freeface 为主；必要时吸纳与其通过“边缘点”相连的 jointface）
    visited = [False] * n
    blocks: List[BlockResult] = []
    current_block_id = 0

    for seed in np.argsort([-getattr(d, 'polygon_area', 0.0) for d in discs]):  # 从面积大到小
        i = int(seed)
        if visited[i] or disc_type[i] == 'undefined':
            continue

        # BFS 在 mutual_graph 上扩展
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

        # 若全是 jointface/undefined，跳过
        if not any(disc_type[idx] == 'freeface' for idx in comp):
            continue

        # step5: 以 freeface 为外壳核心，收集“外壳点云”，并用“边缘点规则”酌情补 jointface
        comp_list = sorted(list(comp))
        freefaces = [idx for idx in comp_list if disc_type[idx] == 'freeface']
        jointfaces = [idx for idx in comp_list if disc_type[idx] == 'jointface']

        # 收集：距离任一 freeface 平面 |dist|<2d 且在其反法向半空间的点
        shell_pts = []
        for fidx in freefaces:
            plane_f, norm_f = disc_planes[fidx], disc_normals[fidx]
            # 对所有邻居（含自身点云）尝试吸纳落入“外壳”条件的点
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

        # --- “边缘点”：同时满足到任一 freeface 与任一 jointface 的 |dist|<2d 的点 ---
        # 这里给出简化实现：若 jointfaces 存在，我们统计与多少 jointface 同时近邻，>0 则纳入补壳点
        if len(jointfaces) > 0:
            jf_planes = [disc_planes[j] for j in jointfaces]
            jf_normals = [disc_normals[j] for j in jointfaces]

            # 为避免 O(N*M*K) 爆炸，先对 shell_pts 下采样（可选）；此处直接用原集合
            # 标注“与多少 jointface 相邻”
            jf_count = np.zeros(shell_pts.shape[0], dtype=int)
            for jp, jn in zip(jf_planes, jf_normals):
                sd = _plane_signed_distance(shell_pts, jp)
                jf_count += (np.abs(sd) < tol).astype(int)

            # 策略：与 >=1 个 jointface 相邻的点当作“边缘点”增强；对于边缘点附近（近邻）再吸纳 jointface 的近邻点以补齐
            edge_mask = jf_count >= 1
            edge_pts = shell_pts[edge_mask]
            if edge_pts.shape[0] > 0:
                # 用 kdtree 在各 jointface 点集中吸纳 |dist|<tol 的点（近似“嵌合面模式”的补壳）
                for j in jointfaces:
                    Pj = disc_points[j]
                    if Pj.shape[0] == 0:
                        continue
                    kdt = cKDTree(Pj)
                    idxs = kdt.query_ball_point(edge_pts, r=tol)
                    picked = np.unique([ii for lst in idxs for ii in lst])
                    if picked.size > 0:
                        shell_pts = np.vstack([shell_pts, Pj[picked]])

        # 网格化：简化为 3D ConvexHull；后续你可替换为 AlphaShape/Poisson 等更贴壳的方法
        try:
            hull = ConvexHull(shell_pts)
            vertices = shell_pts.copy()
            faces = hull.simplices.astype(np.int32)
        except Exception:
            # 点不足或退化，跳过该块
            continue

        # 记录块体
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

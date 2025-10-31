# -*- coding: utf-8 -*-
"""
segment_bak.py — Robust line-segment extraction between elliptical discontinuity disks.
Steps:
 1) 相邻结构面对筛选（以椭圆长半轴为球半径的邻域近似）
 2) 两平面交线求解（方向 + 交线上离原点最近的一点）
 3) 交线与两个椭圆盘分别求交，得到两段参数区间
 4) 两区间取交 → 空间线段 Segment(p1, p2)

依赖：
- numpy / scipy（KDTree）
- 不使用 CGAL；默认单线程，数据量很大时可自行并行分片调用。
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np
from scipy.spatial import cKDTree as KDTree

import networkx as nx
from scipy.spatial import distance
# 参考用户仓库结构
from src import Export  # 用于可选的调试导出

# Discontinuity 类型仅用于注释/可读性；运行时不强制导入
try:
    from src.Discontinuity import Discontinuity, Discontinuitys  # type: ignore
except Exception:
    Discontinuity = object  # noqa
    Discontinuitys = object  # noqa


# ===============================
# 数据结构
# ===============================
class Segment:
    """交线段（与 Export.export_all_segments 兼容）"""
    __slots__ = ("p1", "p2", "dir", "surface_ids", "id", "meta")

    def __init__(self, p1: np.ndarray, p2: np.ndarray,
                 line_dir: np.ndarray,
                 surface_ids: Optional[Tuple[int, int]] = None):
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.p2 = np.asarray(p2, dtype=np.float64)
        self.dir = np.asarray(line_dir, dtype=np.float64)  # 归一化方向
        self.surface_ids = tuple(sorted(surface_ids)) if surface_ids else None
        self.id = None
        self.meta = {}

    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    def as_array(self) -> np.ndarray:
        return np.vstack([self.p1, self.p2])


# ===============================
# 小工具
# ===============================
def _safe_norm(v: np.ndarray, eps: float = 1e-15) -> Tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n


def _any_orthonormal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """给定法向 n（未必单位），在其所在平面上构造一组正交单位向量 (u, v)"""
    n_unit, ln = _safe_norm(np.asarray(n, dtype=np.float64))
    if ln == 0.0:
        # 极端退化：返回 x-y
        return np.array([1., 0., 0.]), np.array([0., 1., 0.])
    # 选一个与 n 不近似平行的轴做叉乘
    if abs(n_unit[0]) < 0.9:
        tmp = np.array([1., 0., 0.])
    else:
        tmp = np.array([0., 1., 0.])
    u = np.cross(n_unit, tmp)
    u, _ = _safe_norm(u)
    v = np.cross(n_unit, u)
    v, _ = _safe_norm(v)
    return u, v


def _get_uv_axes(disc) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取椭圆盘坐标轴单位向量 (u_dir, v_dir)，优先使用 long/short_axis_norm；
    若不存在，尝试由 long/short_axis_vertex 推导；仍无则由法向构造任意正交基。
    """
    # 优先：现成的单位向量
    if getattr(disc, "long_axis_norm", None) is not None and getattr(disc, "short_axis_norm", None) is not None:
        u = np.asarray(disc.long_axis_norm, dtype=np.float64)
        v = np.asarray(disc.short_axis_norm, dtype=np.float64)
        u, _ = _safe_norm(u)
        v, _ = _safe_norm(v)
        return u, v

    # 次选：由两端点推方向
    def vec_from_endpoints(attr_name) -> Optional[np.ndarray]:
        verts = getattr(disc, attr_name, None)
        if verts is None:
            return None
        verts = np.asarray(verts, dtype=np.float64)
        if verts.shape == (2, 3):
            vec = verts[0] - verts[1]
            vec, ln = _safe_norm(vec)
            return vec if ln > 0 else None
        return None

    u = vec_from_endpoints("long_axis_vertex")
    v = vec_from_endpoints("short_axis_vertex")
    if u is not None and v is not None:
        return u, v

    # 兜底：由法向构造 (u, v)
    n = getattr(disc, "disc_normal", getattr(disc, "normal", None))
    if n is None:
        n = np.array([0., 0., 1.])
    return _any_orthonormal_basis(n)


def _line_point_on_two_planes(n1: np.ndarray, p1: np.ndarray,
                              n2: np.ndarray, p2: np.ndarray,
                              d_dir: np.ndarray) -> np.ndarray:
    """
    求两平面交线上“离原点最近”的一点 x：
      解 [n1; n2; d]^T x = [n1·p1, n2·p2, 0]
    更稳定且确定（避免仅用两行约束的漂移）。
    """
    A = np.vstack([n1, n2, d_dir])
    b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0], dtype=np.float64)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 回退最小二乘
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _time_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - (h * 3600 + m * 60)
    return f"{h} h {m} min {s:.2f} sec"


# ===============================
# Step 1: Pair 筛选（球相交近似）
# ===============================
def _candidate_pairs(discs: List[Discontinuity],
                     extension: float = 1.5) -> List[Tuple[int, int]]:
    """
    用 ellip_a 作为半径近似为球，r = ellip_a * extension
    两球相交则认为可能有交线。
    """
    idx_map = []
    centers = []
    radii = []
    for i, d in enumerate(discs):
        if getattr(d, "valid", True) is False:
            continue
        c = getattr(d, "disc_center", None)
        a = getattr(d, "ellip_a", None)
        b = getattr(d, "ellip_b", None)
        if c is None or a is None or b is None:
            continue
        if a <= 0 or b <= 0:
            continue
        centers.append(np.asarray(c, dtype=np.float64))
        radii.append(float(a) * float(extension))
        idx_map.append(i)

    if not centers:
        return []

    centers = np.vstack(centers).astype(np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    tree = KDTree(centers)
    max_r = float(np.max(radii))
    pairs = set()

    for i in range(centers.shape[0]):
        neigh = tree.query_ball_point(centers[i], r=radii[i] + max_r)
        ri = radii[i]
        ci = centers[i]
        for j in neigh:
            if i < j:
                if np.linalg.norm(ci - centers[j]) <= (ri + radii[j]) - 1e-9:
                    pairs.add((idx_map[i], idx_map[j]))

    return sorted(list(pairs))


# ===============================
# Step 2: 平面交线（方向 + 一点）
# ===============================
def _intersect_plane_pairs(disc_pairs: List[Tuple[Discontinuity, Discontinuity]],
                           coplanar_tol: float = 1e-8
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      line_dirs: (M,3) 归一化交线方向
      points_on_line: (M,3) 交线上离原点最近的一点
      mask_valid: (M,) 有效交线（非近似共面）
    """
    M = len(disc_pairs)
    if M == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,), dtype=bool))

    n1s = np.vstack([np.asarray(p[0].disc_normal if getattr(p[0], "disc_normal", None) is not None else p[0].normal,
                                dtype=np.float64) for p in disc_pairs])
    n2s = np.vstack([np.asarray(p[1].disc_normal if getattr(p[1], "disc_normal", None) is not None else p[1].normal,
                                dtype=np.float64) for p in disc_pairs])
    p1s = np.vstack([np.asarray(p[0].disc_center, dtype=np.float64) for p in disc_pairs])
    p2s = np.vstack([np.asarray(p[1].disc_center, dtype=np.float64) for p in disc_pairs])

    d_dirs = np.cross(n1s, n2s)
    norms = np.linalg.norm(d_dirs, axis=1)
    valid = norms >= coplanar_tol

    line_dirs = np.zeros_like(d_dirs)
    line_dirs[valid] = (d_dirs[valid].T / norms[valid]).T  # 归一化

    points = np.full_like(n1s, np.nan, dtype=np.float64)
    idxs = np.where(valid)[0]
    for i in idxs:
        n1 = n1s[i];
        n2 = n2s[i];
        d = line_dirs[i]
        p1 = p1s[i];
        p2 = p2s[i]
        points[i] = _line_point_on_two_planes(n1, p1, n2, p2, d)

    return line_dirs, points, valid


# ===============================
# Step 3: 直线与椭圆盘相交 → 每盘一个参数区间
# ===============================
def _line_ellip_interval(p0: np.ndarray, d: np.ndarray,
                         center: np.ndarray,
                         u_dir: np.ndarray, v_dir: np.ndarray,
                         a: float, b: float,
                         t_eps: float = 1e-9) -> Optional[Tuple[float, float]]:
    """
    求直线 p(t) = p0 + d t 与椭圆 {(x/a)^2 + (y/b)^2 = 1} 的交参数区间 [tmin, tmax]
    其中 x = (p-center)·u_dir，y = (p-center)·v_dir，u_dir/v_dir 为单位向量。
    返回 None 表示不相交或仅切触（过短）。
    """
    rel = p0 - center
    x0 = float(np.dot(rel, u_dir))
    y0 = float(np.dot(rel, v_dir))
    dx = float(np.dot(d, u_dir))
    dy = float(np.dot(d, v_dir))

    # ((x0 + t*dx)/a)^2 + ((y0 + t*dy)/b)^2 = 1
    aa = a * a
    bb = b * b
    A = (dx * dx) / aa + (dy * dy) / bb
    B = 2.0 * ((x0 * dx) / aa + (y0 * dy) / bb)
    C = (x0 * x0) / aa + (y0 * y0) / bb - 1.0

    if abs(A) < 1e-18:
        # 直线与椭圆边界平行：当 C<=0 时直线在椭圆内（“无界”）→此处视为不计
        return None

    D = B * B - 4.0 * A * C
    if D < 0.0:
        return None

    sqrtD = math.sqrt(max(D, 0.0))
    t1 = (-B - sqrtD) / (2.0 * A)
    t2 = (-B + sqrtD) / (2.0 * A)
    lo = min(t1, t2)
    hi = max(t1, t2)

    if hi - lo <= t_eps:
        return None
    return lo, hi


def _build_intervals_for_pairs(disc_pairs: List[Tuple[Discontinuity, Discontinuity]],
                               line_dirs: np.ndarray,
                               points_on_line: np.ndarray,
                               mask_valid: np.ndarray,
                               t_eps: float = 1e-9
                               ) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    为每个有效 pair 构造两张椭圆盘的交参数区间。
    仅当两个区间都存在时，记为有效 pair。
    返回：pair_idx -> ((tmin1,tmax1), (tmin2,tmax2))
    """
    valid_pairs: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    for pair_idx, (d1, d2) in enumerate(disc_pairs):
        if not mask_valid[pair_idx]:
            continue

        p0 = points_on_line[pair_idx]
        dir_vec = line_dirs[pair_idx]

        # 盘1
        c1 = np.asarray(d1.disc_center, dtype=np.float64)
        a1 = float(d1.ellip_a)
        b1 = float(d1.ellip_b)
        u1, v1 = _get_uv_axes(d1)
        inter1 = _line_ellip_interval(p0, dir_vec, c1, u1, v1, a1, b1, t_eps)

        # 盘2
        c2 = np.asarray(d2.disc_center, dtype=np.float64)
        a2 = float(d2.ellip_a)
        b2 = float(d2.ellip_b)
        u2, v2 = _get_uv_axes(d2)
        inter2 = _line_ellip_interval(p0, dir_vec, c2, u2, v2, a2, b2, t_eps)

        if inter1 is None or inter2 is None:
            continue

        valid_pairs[pair_idx] = (inter1, inter2)

    return valid_pairs


# ===============================
# Step 4: 区间相交 → 空间线段
# ===============================
def _intervals_to_segments(valid_intervals: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]],
                           disc_pairs: List[Tuple[Discontinuity, Discontinuity]],
                           line_dirs: np.ndarray,
                           points_on_line: np.ndarray,
                           length_eps: float = 1e-6) -> List[Segment]:
    out: List[Segment] = []

    for pair_idx, ((t1_lo, t1_hi), (t2_lo, t2_hi)) in valid_intervals.items():
        lo = max(t1_lo, t2_lo)
        hi = min(t1_hi, t2_hi)
        if hi - lo <= float(length_eps):
            continue

        p0 = points_on_line[pair_idx]
        d = line_dirs[pair_idx]
        a = p0 + lo * d
        b = p0 + hi * d

        # 附带 surface_ids（使用 cluster_id 以便后续追踪）
        s1 = getattr(disc_pairs[pair_idx][0], "cluster_id", None)
        s2 = getattr(disc_pairs[pair_idx][1], "cluster_id", None)
        seg = Segment(a, b, d, (s1, s2))
        out.append(seg)

    return out


# ===============================
# 对外主入口
# ===============================
def get_segments(discontinuitys: Discontinuitys,
                 extension: float = 2,
                 coplanar_tol: float = 1e-8,
                 t_eps: float = 1e-9,
                 length_eps: float = 1e-6,
                 verbose: bool = True,
                 debug_export_line_dir: Optional[str] = None
                 ) -> List[Segment]:
    """
    主流程：
      - extension：邻域放大系数（用 ellip_a*extension 做球半径筛 Pair）
      - coplanar_tol：判断近似共面的阈值（交线方向范数下限）
      - t_eps：判别“切触/极短区间”的阈值
      - length_eps：过滤极短线段
      - debug_export_line_dir：若给定路径，则额外导出交线方向（每条线段 ±3m）

    返回：Segment 列表（含 p1/p2/dir/surface_ids）
    """
    t0 = time.perf_counter()
    discs: List[Discontinuity] = list(getattr(discontinuitys, "discontinuitys", []))

    if verbose:
        print(f"[segments] total discontinuities: {len(discs)}")

    # Step 1
    t = time.perf_counter()
    pair_indices = _candidate_pairs(discs, extension=extension)
    disc_pairs = [(discs[i], discs[j]) for (i, j) in pair_indices]
    if verbose:
        print(f"[pairs] candidates: {len(disc_pairs)} | {_time_hms(time.perf_counter() - t)}")

    # Step 2
    t = time.perf_counter()
    line_dirs, points_on_line, mask_valid = _intersect_plane_pairs(disc_pairs, coplanar_tol=coplanar_tol)
    if verbose:
        print(f"[planes] valid intersections: {int(mask_valid.sum())}, "
              f"coplanar/invalid: {int(len(mask_valid) - mask_valid.sum())} | "
              f"{_time_hms(time.perf_counter() - t)}")

    # 可选：导出交线方向（便于检查几何）
    if debug_export_line_dir is not None and line_dirs.shape[0] > 0:
        starts = points_on_line + 3.0 * line_dirs
        ends = points_on_line - 3.0 * line_dirs
        pts = np.vstack([starts, ends]).astype(np.float32)
        edges = np.c_[np.arange(0, len(starts)) * 2, np.arange(0, len(starts)) * 2 + 1].astype(np.int32)
        Export.write_ply_with_edges(debug_export_line_dir, pts, edges)
        if verbose:
            print(f"[debug] exported line dirs to: {debug_export_line_dir}")

    # Step 3
    t = time.perf_counter()
    valid_intervals = _build_intervals_for_pairs(disc_pairs, line_dirs, points_on_line, mask_valid, t_eps=t_eps)
    if verbose:
        print(f"[ellipse] pairs with two valid intervals: {len(valid_intervals)} | "
              f"{_time_hms(time.perf_counter() - t)}")

    # Step 4
    t = time.perf_counter()
    segments = _intervals_to_segments(valid_intervals, disc_pairs, line_dirs, points_on_line, length_eps=length_eps)
    if verbose:
        print(f"[segments] produced: {len(segments)} | {_time_hms(time.perf_counter() - t)}")

    if verbose:
        print(f"[total] {_time_hms(time.perf_counter() - t0)}")
    return segments


# ===============================
# 建立节点（node）与边（edge）图 graph
# ===============================
import numpy as np
from typing import List, Optional


def bounding_box(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    计算给定两点的包围盒，返回 [x_min, x_max, y_min, y_max]
    """
    return np.array([min(p1[0], p2[0]), max(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[1], p2[1])])


def is_bounding_box_intersect(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    判断两个包围盒是否相交。
    box1, box2 格式为 [x_min, x_max, y_min, y_max]
    """
    return not (box1[1] < box2[0] or box1[0] > box2[1] or box1[3] < box2[2] or box1[2] > box2[3])


def find_intersection_point(seg1: Segment, seg2: Segment, tolerance=1e-3) -> Optional[np.ndarray]:
    """
    判断两个线段（seg1, seg2）是否相交，如果相交则返回交点，否则返回 None。
    先通过包围盒做预筛选，再计算交点。
    """

    p1, p2 = seg1.p1, seg1.p2
    p3, p4 = seg2.p1, seg2.p2

    #  计算交点
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(denom) < tolerance:  # 线段平行
        return None

    # 交点的参数 t1（参数方程）
    t1 = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    t2 = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom

    # 判断交点是否在两个线段范围内
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        intersection = p1 + t1 * d1
        return intersection
    return None


def build_graph(segments: List[Segment], tolerance=1e-3) -> nx.Graph:
    """
    根据交点构建图。节点为交点，边为相交的线段。
    这里优化了包围盒计算，先通过包围盒快速筛选，再进行精确计算交点。
    """
    G = nx.Graph()

    # 用来存储已存在的交点
    points_map = {}

    def add_point(point):
        # 将交点加入节点
        point_tuple = tuple(np.round(point, decimals=6))  # 处理浮点数精度问题
        if point_tuple not in points_map:
            node_id = len(points_map)
            points_map[point_tuple] = node_id
            G.add_node(node_id, point=point)
        return points_map[point_tuple]

    # 计算所有线段的包围盒
    boxes = np.array([bounding_box(seg.p1, seg.p2) for seg in segments])

    # 构建边
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue

            # 使用包围盒做快速筛选
            if not is_bounding_box_intersect(boxes[i], boxes[j]):
                continue

            intersection = find_intersection_point(seg1, seg2, tolerance)
            if intersection is not None:
                # print("has node")
                # 获取交点的节点编号
                node1 = add_point(seg1.p1)
                node2 = add_point(seg1.p2)
                node3 = add_point(seg2.p1)
                node4 = add_point(seg2.p2)

                # 将边添加到图中
                G.add_edge(node1, node3, segment=(seg1, seg2))
                G.add_edge(node1, node4, segment=(seg1, seg2))
                G.add_edge(node2, node3, segment=(seg1, seg2))
                G.add_edge(node2, node4, segment=(seg1, seg2))

    # 修剪非环节点（毛刺）
    G = prune_isolated_nodes(G)

    return G


def prune_isolated_nodes(G: nx.Graph) -> nx.Graph:
    """
    修剪掉度数为零的孤立节点（毛刺），保留图中有实际连接的节点。
    """
    # 获取所有度数大于 0 的节点
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]

    # 从图中移除这些孤立节点
    G.remove_nodes_from(nodes_to_remove)

    return G


# ===============================
# 寻找潜在的blocks：对于构建好的图，使用图算法（例如深度优先搜索 DFS 或连通分量算法）来寻找 "blocks"，即具有一定连通性的区域。
# ===============================
def find_blocks(graph: nx.Graph) -> List[List[int]]:
    """
    查找潜在的blocks，blocks为图的连通分量。
    """
    return list(nx.connected_components(graph))


def visualize_graph(G: nx.Graph):
    """
    可视化图，展示节点和边（基于 NetworkX 提供的可视化工具）。
    """
    # 使用 spring layout 布局节点
    pos = nx.spring_layout(G, seed=42)  # 用一个固定的seed确保每次布局一致
    plt.figure(figsize=(8, 8))  # 调整图的大小
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, edge_color="gray", width=1)
    plt.title("Graph Visualization")
    plt.show()


# ===============================
# 可选：简单自检（示例）
# ===============================
if __name__ == "__main__":
    # 这里放一个空壳，避免误运行时报错。
    # 实际集成时由上层调用 get_segments(...)。
    print("segment_bak.py loaded. Use get_segments(discontinuitys, ...).")

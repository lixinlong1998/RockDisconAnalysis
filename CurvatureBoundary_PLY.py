# CurvatureBoundary_PLY.py
# 功能：基于 PCA 曲率的点云边界检测（读PLY → 计算曲率 → 导出边界/非边界）
# 依赖：numpy, scipy, plyfile

import os
import time
import argparse
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

'''
# 1) 默认参数，在输入同目录输出：
python CurvatureBoundary_PLY.py `
--input "E:\Database\_RockPoints\TSDK_Rockfall_RegularClip\TSDK_Rockfall_13_P1_ORG.ply" `
--out_dir "D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\TSDK_Rockfall_13_P1_ORG_workspace"
'''


# ---------- PLY I/O ----------
def read_ply_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 PLY，返回 (points[N,3], vertex_struct_array[N])。"""
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError("PLY 中没有 'vertex' 元素。")
    v = ply["vertex"].data  # structured array
    names = v.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("vertex 中缺少 x/y/z 坐标。")
    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    return pts, v


def write_ply_vertices(path: str, vertex_struct_array: np.ndarray):
    """写出仅含 vertex 的 PLY；尽量保留原字段。"""
    out = np.array(vertex_struct_array, copy=True)
    el = PlyElement.describe(out, "vertex")
    PlyData([el], text=False).write(path)


def append_fields(vertex_struct: np.ndarray,
                  fields: dict) -> np.ndarray:
    """
    向 structured array 追加/覆盖若干字段（float/uint8 等），返回新数组。
    fields: {name: np.ndarray(shape=(N,), dtype)}
    """
    v = vertex_struct
    N = v.shape[0]
    names = v.dtype.names or ()
    # 若所有字段都已存在，则覆盖
    if all(n in names for n in fields.keys()):
        v2 = v.copy()
        for k, arr in fields.items():
            v2[k] = arr.astype(v2.dtype.fields[k][0]) if k in v2.dtype.fields else arr
        return v2

    # 否则扩展 dtype
    old_descr = v.dtype.descr
    add_descr = []
    for k, arr in fields.items():
        if k in names:
            continue
        # 统一用小端类型：float32 -> '<f4'，uint8 -> 'u1'
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            add_descr.append((k, "<f4"))
        elif arr.dtype == np.uint8:
            add_descr.append((k, "u1"))
        elif arr.dtype == np.bool_:
            add_descr.append((k, "u1"))  # bool 转存为 0/1
        else:
            # 默认用 float32
            add_descr.append((k, "<f4"))

    new_descr = old_descr + add_descr
    v_new = np.empty(N, dtype=new_descr)
    # 复制旧字段
    for name in v.dtype.names:
        v_new[name] = v[name]
    # 填充新增/覆盖字段
    for k, arr in fields.items():
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        if k in v.dtype.names:
            v_new[k] = arr.astype(v_new.dtype.fields[k][0])
        else:
            v_new[k] = arr.astype(v_new.dtype.fields[k][0])
    return v_new


# ---------- PCA 曲率 ----------
def _pca_eigs(neigh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对邻域点 (m,3) 做 PCA，返回 (eigvals升序(3,), eigvecs列向量(3,3))
    """
    X = neigh - neigh.mean(axis=0, keepdims=True)
    C = X.T @ X  # 未除以(m-1)的协方差比例即可；仅相对量
    w, v = np.linalg.eigh(C)  # w升序
    return w, v


def compute_curvature(points: np.ndarray,
                      k: Optional[int] = 30,
                      radius: Optional[float] = None,
                      min_neighbors: int = 10,
                      sample_cap: Optional[int] = None,
                      return_normals: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    计算每个点的 PCA 曲率：lam0 / (lam0+lam1+lam2)
    - k 与 radius 至少给一个（默认用 k）
    - min_neighbors: 邻域最少点数，不足则曲率记为0（或跳过）
    - sample_cap: 邻域点过多时随机下采样上限，加速
    - return_normals: 若 True，同步返回法向（最小特征值对应特征向量）
    返回:
      curvature: (N,)
      normals: (N,3) 或 None
    """
    N = points.shape[0]
    curv = np.zeros(N, dtype=np.float32)
    normals = np.zeros((N, 3), dtype=np.float32) if return_normals else None

    tree = cKDTree(points)

    if radius is not None:
        neighborhoods = tree.query_ball_point(points, radius)
        use_radius = True
    else:
        use_radius = False
        # cKDTree.query 的 k 必含自身
        k = int(k or 30)
        dists, idxs = tree.query(points, k=k)

    eps = 1e-12

    for i in range(N):
        if use_radius:
            idx = neighborhoods[i]
        else:
            idx = idxs[i].tolist()

        m = len(idx)
        if m < min_neighbors:
            curv[i] = 0.0
            if return_normals:
                normals[i] = np.array([0, 0, 1], dtype=np.float32)
            continue

        if sample_cap is not None and m > sample_cap:
            sel = np.random.choice(m, size=sample_cap, replace=False)
            idx = [idx[s] for s in sel]
            m = len(idx)

        neigh = points[idx]
        w, v = _pca_eigs(neigh)  # w升序
        s = float(w.sum()) + eps
        c = float(w[0]) / s
        curv[i] = c
        if return_normals:
            n = v[:, 0]
            n /= (np.linalg.norm(n) + 1e-12)
            normals[i] = n.astype(np.float32)

    return curv, normals


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(
        description="基于 PCA 曲率 (λ0/∑λ) 的点云边界检测（读PLY→导出边界/非边界）")
    parser.add_argument("--input", type=str, help="输入 .ply")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认同输入）")
    parser.add_argument("--prefix", type=str, default=None, help="输出文件名前缀（默认用输入名）")

    # 邻域设置（k 与 radius 二选一，默认 kNN）
    parser.add_argument("--k", type=int, default=30, help="PCA 邻域 k（默认 30）")
    parser.add_argument("--radius", type=float, default=0.3, help="PCA 邻域半径（提供则优先生效）")
    parser.add_argument("--min_neighbors", type=int, default=10, help="最小邻域点数")
    parser.add_argument("--sample_cap", type=int, default=256, help="邻域下采样上限以提速")

    # 判定阈值
    parser.add_argument("--thr", type=float, default=0.02,
                        help="曲率阈值，>thr 视为边界（默认 0.06）")  # 越小提取的边界点越多,越大对边界点越不敏感

    # 额外输出选项
    parser.add_argument("--write_single_with_fields", action="store_true",
                        help="同时写一个带 curvature/boundary 字段的单文件便于上色查看")
    parser.add_argument("--write_normals", action="store_true",
                        help="在单文件里追加 PCA 法向 (nx,ny,nz) 字段（用于可视化）")

    args = parser.parse_args()

    in_path = args.input
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(in_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    prefix = args.prefix or stem

    t0 = time.perf_counter()
    print(f"[I] 读取: {in_path}")
    points, v = read_ply_points(in_path)
    N = points.shape[0]
    print(f"[I] 点数: {N:,d}")

    # 计算曲率（及可选法向）
    t1 = time.perf_counter()
    curv, nrm = compute_curvature(
        points,
        k=None if args.radius is not None else args.k,
        radius=args.radius,
        min_neighbors=args.min_neighbors,
        sample_cap=args.sample_cap,
        return_normals=args.write_single_with_fields and args.write_normals
    )
    t2 = time.perf_counter()
    print(f"[I] 曲率已计算，用时 {(t2 - t1):.2f}s | "
          f"统计: min={curv.min():.4f}, p50={np.median(curv):.4f}, p95={np.percentile(curv, 95):.4f}, max={curv.max():.4f}")

    # 阈值分割
    thr = float(args.thr)
    boundary_mask = (curv > thr)
    n_edge = int(boundary_mask.sum())
    print(f"[I] 边界点: {n_edge:,d} / {N:,d} ({n_edge / max(N, 1):.2%}) | 阈值: {thr}")

    # 写出分离的两个 PLY
    out_edge = os.path.join(out_dir, f"{prefix}_curv_edge.ply")
    out_non = os.path.join(out_dir, f"{prefix}_curv_nonedge.ply")
    write_ply_vertices(out_edge, v[boundary_mask])
    write_ply_vertices(out_non, v[~boundary_mask])
    print(f"[I] 已写出:\n  - {out_edge} ({n_edge:,d})\n  - {out_non} ({N - n_edge:,d})")

    # 可选：写出单文件，带 curvature/boundary/(nx,ny,nz)
    if args.write_single_with_fields:
        fields = {
            "curvature": curv.astype(np.float32),
            "boundary": boundary_mask.astype(np.uint8),
        }
        if args.write_normals and nrm is not None:
            fields.update({
                "nx": nrm[:, 0].astype(np.float32),
                "ny": nrm[:, 1].astype(np.float32),
                "nz": nrm[:, 2].astype(np.float32),
            })
        v_all = append_fields(v, fields)
        out_all = os.path.join(out_dir, f"{prefix}_curv_all.ply")
        write_ply_vertices(out_all, v_all)
        print(f"[I] 也已写出带字段的单文件: {out_all}")

    t3 = time.perf_counter()
    print(f"[I] 总用时: {(t3 - t0):.2f}s")


if __name__ == "__main__":
    main()

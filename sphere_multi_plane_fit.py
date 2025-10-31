import argparse
import os
import math
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import open3d as o3d

'''
使用下面的脚本:
python sphere_multi_plane_fit.py `
--input "D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\TSDK_Rockfall_13_P1_ORG_facets_Kd_E0.3A10\TSDK_Rockfall_13_P1_ORG_facets_points.ply" `
--out "D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\TSDK_Rockfall_13_P1_ORG_facets_Kd_E0.3A10\sphere_multi_plane_fit" `
--d 1 --ransac_tol 0.01

# 或者用半径指定（脚本会自动反推 d = r*√2 ）
python sphere_multi_plane_fit.py --input your_cloud.ply --out out_dir --r 0.707 --ransac_tol 0.01
'''
# ---------- 几何与模型选择工具 ----------

def fit_plane_svd(P):
    """用 SVD 拟合平面 ax+by+cz+d=0，返回 (n, d0, residuals)
       n 为单位法向，d0 为偏置，使得 n·x + d0 = 0"""
    centroid = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - centroid, full_matrices=False)
    n = Vt[-1, :]
    n = n / np.linalg.norm(n)
    d0 = -np.dot(n, centroid)
    # 点到平面有符号距离
    dist = P @ n + d0
    return n, d0, np.abs(dist)


def ransac_plane(P, tol=0.01, max_iters=500, random_state=None):
    """RANSAC 找到平面内点索引与模型"""
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_state)

    n_points = P.shape[0]
    if n_points < 3:
        return None

    best_inliers = None
    best_model = None
    for _ in range(max_iters):
        # 随机取3点，确保不共线
        idx = rng.choice(n_points, size=3, replace=False)
        A, B, C = P[idx]
        v1 = B - A
        v2 = C - A
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-9:
            continue
        n = n / norm_n
        d0 = -np.dot(n, A)
        dist = np.abs(P @ n + d0)
        inliers = np.where(dist <= tol)[0]
        if best_inliers is None or inliers.size > best_inliers.size:
            best_inliers = inliers
            best_model = (n, d0)
    if best_inliers is None or best_inliers.size < 3:
        return None
    # 对内点再精拟合
    n_refined, d0_refined, _ = fit_plane_svd(P[best_inliers])
    return {
        "model": (n_refined, d0_refined),
        "inliers": best_inliers
    }


def multi_plane_ransac(P, k=1, tol=0.01, max_iters=500, min_cluster=50, random_state=None):
    """顺序剥离的多平面 RANSAC：返回 k 个平面（可能不足 k），以及每个点的簇标签（-1=噪声）"""
    labels = -np.ones(P.shape[0], dtype=int)
    remaining_idx = np.arange(P.shape[0])
    planes = []
    for ci in range(k):
        if remaining_idx.size < min_cluster:
            break
        subP = P[remaining_idx]
        res = ransac_plane(subP, tol=tol, max_iters=max_iters, random_state=random_state)
        if res is None:
            break
        inliers_local = res["inliers"]
        if inliers_local.size < min_cluster:
            break
        # 记录该平面
        n, d0 = res["model"]
        # 精配后再算该子集的残差，以便二次过滤
        dist_full = np.abs(subP @ n + d0)
        inliers_local = np.where(dist_full <= tol)[0]
        if inliers_local.size < min_cluster:
            break
        global_inliers = remaining_idx[inliers_local]
        labels[global_inliers] = ci
        planes.append((n, d0))
        # 更新剩余集合
        mask = np.ones(remaining_idx.size, dtype=bool)
        mask[inliers_local] = False
        remaining_idx = remaining_idx[mask]
    return planes, labels


def model_bic(P, planes, labels):
    """计算 BIC 作为模型优劣（越小越好）
       近似设每个平面 3 个自由度，噪声点也计入 RSS"""
    n = P.shape[0]
    if n == 0:
        return np.inf
    # 汇总 RSS（平方残差）
    rss = 0.0
    unique_lbls = [l for l in np.unique(labels) if l >= 0]
    for ci in unique_lbls:
        nvec, d0 = planes[ci]
        Pi = P[labels == ci]
        if Pi.size == 0:
            continue
        dist = Pi @ nvec + d0
        rss += np.sum(dist * dist)
    # 噪声点：按“没有解释”的残差，给个惩罚（等效大残差）
    noise_cnt = np.sum(labels < 0)
    if noise_cnt > 0:
        # 用整体点云到全体点的最近平面距离的均值估计一个噪声残差
        # 这里简化为：噪声每点加上 tol^2 的等效项，以鼓励解释更多点
        tol = max(1e-6, np.percentile(np.linalg.norm(P - P.mean(0), axis=1), 5) * 1e-3)
        rss += noise_cnt * (tol ** 2)
    p = max(1, 3 * len(unique_lbls))  # 参数数(粗略)
    bic = n * np.log(max(rss / n, 1e-12)) + p * np.log(n)
    return bic


# ---------- 颜色工具 ----------

def color_palette():
    # 预设一组高区分度颜色（RGB 0..1）
    return np.array([
        [0.90, 0.10, 0.10],  # 红
        [0.10, 0.60, 0.95],  # 蓝
        [0.10, 0.75, 0.25],  # 绿
        [0.95, 0.80, 0.10],  # 黄
        [0.65, 0.30, 0.95],  # 紫
        [0.95, 0.50, 0.15],  # 橙
        [0.15, 0.85, 0.85],  # 青
        [0.60, 0.60, 0.60],  # 灰
    ])


def class_color_map():
    # A/B/C 类别颜色（球心）
    return {
        'A': np.array([0.20, 0.80, 0.20]),  # 绿
        'B': np.array([0.20, 0.50, 0.95]),  # 蓝
        'C': np.array([0.90, 0.30, 0.20]),  # 红
        'E': np.array([0.75, 0.75, 0.75])  # 空/不足
    }


# ---------- 主流程 ----------

def build_grid_centers(bmin, bmax, d=None, r=None):
    if d is None and r is None:
        raise ValueError("必须提供 d 或 r 之一")
    if d is None:
        d = r * math.sqrt(2.0)
    if r is None:
        r = d * math.sqrt(2.0) / 2.0
    # 根据 d 生成规则网格；为避免漏边界，向外扩一点
    pad = r * 0.5
    starts = bmin - pad
    ends = bmax + pad
    xs = np.arange(starts[0], ends[0] + 1e-9, d)
    ys = np.arange(starts[1], ends[1] + 1e-9, d)
    zs = np.arange(starts[2], ends[2] + 1e-9, d)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')
    centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return centers, d, r


def save_ply_points(filepath, points, colors=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(filepath, pc, write_ascii=True)


def main():
    ap = argparse.ArgumentParser(description="球体内 1/2/3 平面拟合 + 结果可视化导出")
    ap.add_argument("--input", required=True, help="输入点云 PLY（只需 XYZ，含颜色也可）")
    ap.add_argument("--out", required=True, help="输出文件夹")
    ap.add_argument("--d", type=float, default=None, help="球心间距 d（若不给则用 r 推算）")
    ap.add_argument("--r", type=float, default=None, help="球半径 r（若不给则用 d 推算）")
    ap.add_argument("--ransac_tol", type=float, default=0.01, help="RANSAC 内点阈值")
    ap.add_argument("--max_iters", type=int, default=500, help="RANSAC 最大迭代")
    ap.add_argument("--min_pts", type=int, default=200, help="每球最少点数（不足则标记为空）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sphere_dir = os.path.join(args.out, "spheres")
    os.makedirs(sphere_dir, exist_ok=True)

    # 读取点云
    pc = o3d.io.read_point_cloud(args.input)
    if len(pc.points) == 0:
        raise RuntimeError("点云为空")
    P = np.asarray(pc.points, dtype=np.float64)
    # KD-Tree
    kdt = cKDTree(P)

    # 边界盒 & 网格球心
    bmin = P.min(axis=0)
    bmax = P.max(axis=0)
    centers, d, r = build_grid_centers(bmin, bmax, d=args.d, r=args.r)
    print(f"[INFO] 生成球心 {centers.shape[0]} 个；d={d:.4f}, r={r:.4f}")

    # 遍历球
    cls_color = class_color_map()
    palette = color_palette()

    center_points = []
    center_colors = []
    meta = []

    for idx in tqdm(range(centers.shape[0]), desc="处理球体"):
        c = centers[idx]
        ids = kdt.query_ball_point(c, r)
        if len(ids) < args.min_pts:
            center_points.append(c)
            center_colors.append(cls_color['E'])
            meta.append(("E", 0))
            continue
        Pi = P[ids]

        # 尝试 k=1,2,3
        best = None
        for k in (1, 2, 3):
            planes, labels = multi_plane_ransac(
                Pi, k=k, tol=args.ransac_tol, max_iters=args.max_iters,
                min_cluster=max(20, args.min_pts // (k + 1)),
                random_state=args.seed + k + idx
            )
            bic = model_bic(Pi, planes, labels)
            inliers_count = np.sum(labels >= 0)
            cand = {"k": len(planes), "labels": labels, "planes": planes, "bic": bic, "inliers": inliers_count}
            if best is None or (cand["k"] > best["k"] and cand["bic"] <= best["bic"] * 1.05) or (
                    cand["bic"] < best["bic"]):
                best = cand

        # 分类 A/B/C（以拟合到的平面数量）
        k_sel = best["k"]
        if k_sel <= 0:
            cat = 'E'
            center_color = cls_color['E']
        elif k_sel == 1:
            cat = 'A'
            center_color = cls_color['A']
        elif k_sel == 2:
            cat = 'B'
            center_color = cls_color['B']
        else:
            cat = 'C'
            center_color = cls_color['C']

        center_points.append(c)
        center_colors.append(center_color)
        meta.append((cat, k_sel))

        # 为该球导出 PLY：点按 labels 分色，噪声灰色
        labels = best["labels"]
        colors = np.zeros((Pi.shape[0], 3), dtype=float)
        colors[:] = np.array([0.75, 0.75, 0.75])  # 噪声灰
        for ci in range(min(3, k_sel)):
            col = palette[ci % len(palette)]
            colors[labels == ci] = col
        out_path = os.path.join(sphere_dir, f"sphere_{idx:06d}_{cat}_k{k_sel}.ply")
        save_ply_points(out_path, Pi, colors)

    # 导出球心总览
    centers_arr = np.vstack(center_points) if center_points else centers
    center_cols = np.vstack(center_colors) if center_colors else np.tile(np.array([0.9, 0.9, 0.9]),
                                                                         (centers.shape[0], 1))
    save_ply_points(os.path.join(args.out, "centers.ply"), centers_arr, center_cols)

    # 简要统计
    cats = [m[0] for m in meta]
    print("[SUMMARY] A(单)={}, B(双)={}, C(三)={}, E(空/不足)={}".format(
        cats.count('A'), cats.count('B'), cats.count('C'), cats.count('E')
    ))
    print(f"[DONE] 输出目录：{args.out}")


if __name__ == "__main__":
    main()

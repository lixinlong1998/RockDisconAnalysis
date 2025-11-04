# -*- coding: utf-8 -*-
"""
Multi-planes point clouds with small normal perturbation + iterative RANSAC.
- Cube: [0, CUBE_SIZE]^3
- Two/three adjacent planes come from {x=0, y=0, z=0}, then each plane is slightly tilted.
- Output: 13 datasets (.npz) + ransac_summary.csv
- Visualization: matplotlib (wireframe plane), open3d (patch mesh) if installed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =========================
# 参数区(直接修改)
# =========================
CUBE_SIZE: float = 10.0  # 立方体边长(m)
TOTAL_POINTS: int = 20000  # 每组总点数(含离群点)
GAUSS_NOISE_SIGMA: float = 0.02  # 平面点噪声σ(m)
OUTLIER_FRAC: float = 0.02  # 离群点比例
RANDOM_ANGLE_DEG: float = 5.0  # 真实面法向扰动±(度)

RANSAC_DIST_THRESH: float = 0.03  # RANSAC内点阈值(m)
RANSAC_MAX_ITERS: int = 1200  # RANSAC迭代次数
RANSAC_MIN_INLIER_FRAC: float = 0.02  # 单次模型最小内点比例(相对当前剩余点)
STOP_THRESHOLD_NUMBER: int = 1000  # 剩余点数≤该值停止

RNG_SEED: int = 123
OUT_DIR = Path("./plane_ransac_experiment")

# 可选可视化依赖
try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False
try:
    import open3d as o3d

    HAS_O3D = True
except Exception:
    HAS_O3D = False


# =========================
# 线性代数与几何
# =========================
def rot_matrix(axis: np.ndarray, ang: float) -> np.ndarray:
    """
    功能: Rodrigues公式构造3x3旋转矩阵
    输入:
        axis (np.ndarray[3]): 旋转轴(可非单位)
        ang (float): 旋转角(弧度)
    输出:
        R (np.ndarray[3,3]): 旋转矩阵
    """
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a;
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], float)


def rotate_about_centroid(pts: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """
    功能: 绕点集质心旋转整片点
    输入:
        pts (np.ndarray[N,3]): 点云
        axis (np.ndarray[3]): 旋转轴
        ang (float): 弧度
    输出:
        pts2 (np.ndarray[N,3]): 旋转后
    """
    if pts.size == 0: return pts
    R = rot_matrix(axis, ang)
    c = pts.mean(axis=0)
    return (pts - c) @ R.T + c


def to_cube(pts: np.ndarray) -> np.ndarray:
    """功能: 裁剪到[0,CUBE_SIZE]^3; 输入/输出同shape"""
    return np.clip(pts, 0.0, CUBE_SIZE)


# =========================
# 平面采样与法向扰动
# =========================
def samp_xy_z(zv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    功能: 在z=zv平面采样n点+高斯噪声
    输入:
        zv (float), n (int), rng (np.random.Generator)
    输出:
        pts (np.ndarray[n,3])
    """
    xy = rng.uniform(0, CUBE_SIZE, (n, 2))
    z = np.full((n, 1), zv)
    pts = np.concatenate([xy, z], 1)
    pts += rng.normal(0, GAUSS_NOISE_SIGMA, pts.shape)
    return to_cube(pts)


def samp_yz_x(xv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """功能同上：x=xv平面"""
    yz = rng.uniform(0, CUBE_SIZE, (n, 2))
    x = np.full((n, 1), xv)
    pts = np.concatenate([x, yz], 1)
    pts += rng.normal(0, GAUSS_NOISE_SIGMA, pts.shape)
    return to_cube(pts)


def samp_xz_y(yv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """功能同上：y=yv平面"""
    xz = rng.uniform(0, CUBE_SIZE, (n, 2))
    y = np.full((n, 1), yv)
    pts = np.concatenate([xz[:, [0]], y, xz[:, [1]]], 1)
    pts += rng.normal(0, GAUSS_NOISE_SIGMA, pts.shape)
    return to_cube(pts)


def add_out(n: int, rng: np.random.Generator) -> np.ndarray:
    """功能: 立方体内均匀离群点; 输入: n; 输出: (n,3)"""
    return rng.uniform(0, CUBE_SIZE, (n, 3))


def tilt_plane(pts: np.ndarray, base_n: np.ndarray, max_deg: float,
               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    功能: 对一片平面点整体做小角度旋转, 返回旋转后点与真实法向
    输入:
        pts (np.ndarray[N,3]): 平面点
        base_n (np.ndarray[3]): 基准法向(如[1,0,0])
        max_deg (float): ±扰动幅度(度)
        rng (np.random.Generator): 随机源
    输出:
        pts2 (np.ndarray[N,3]), n_true (np.ndarray[3])
    """
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    axis = rng.normal(0, 1, 3);
    axis /= (np.linalg.norm(axis) + 1e-12)
    R = rot_matrix(axis, ang)
    pts2 = to_cube(rotate_about_centroid(pts, axis, ang))
    n_true = R @ (base_n / (np.linalg.norm(base_n) + 1e-12))
    n_true /= (np.linalg.norm(n_true) + 1e-12)
    return pts2, n_true


# =========================
# 数据集构造
# =========================
def ds_single(total: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    功能: 单平面(z≈0扰动)+离群
    输入:
        total (int): 总点数
        rng (Generator)
    输出:
        pts (N,3), lbl (N,) {0:plane,1:out}, meta (dict)
    """
    n_out = int(total * OUTLIER_FRAC);
    n_in = total - n_out
    p0 = samp_xy_z(0.0, n_in, rng)
    p1, n1 = tilt_plane(p0, np.array([0, 0, 1.0]), RANDOM_ANGLE_DEG, rng)
    o = add_out(n_out, rng)
    pts = np.vstack([p1, o]);
    lbl = np.concatenate([np.zeros(n_in, int), np.ones(n_out, int)])
    meta = {"truth_planes": [{"name": "single(z≈0)", "normal": n1.tolist()}]}
    return pts, lbl, meta


def ds_two(pa: float, total: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    功能: 两相邻面(x≈0,y≈0)+离群; 点数占比=pa:(1-pa)
    输入:
        pa (float), total (int), rng
    输出:
        pts (N,3), lbl (N,) {0:A,1:B,2:out}, meta
    """
    n_out = int(total * OUTLIER_FRAC);
    n_in = total - n_out
    n_a = int(n_in * pa);
    n_b = n_in - n_a
    A0 = samp_yz_x(0.0, n_a, rng);
    B0 = samp_xz_y(0.0, n_b, rng)
    A, nA = tilt_plane(A0, np.array([1.0, 0, 0]), RANDOM_ANGLE_DEG, rng)
    B, nB = tilt_plane(B0, np.array([0, 1.0, 0]), RANDOM_ANGLE_DEG, rng)
    o = add_out(n_out, rng)
    pts = np.vstack([A, B, o])
    lbl = np.concatenate([np.full(n_a, 0), np.full(n_b, 1), np.full(n_out, 2)])
    meta = {"truth_planes": [{"name": "A(≈x=0)", "normal": nA.tolist()},
                             {"name": "B(≈y=0)", "normal": nB.tolist()}],
            "ratios": {"A": pa, "B": 1 - pa}}
    return pts, lbl, meta


def ds_three(pa: float, pb: float, pc: float, total: int,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    功能: 三相邻面(立方体一角 x≈0,y≈0,z≈0)+离群; 占比=pa,pb,pc(和=1)
    输入:
        pa,pb,pc (float), total (int), rng
    输出:
        pts (N,3), lbl (N,) {0:A,1:B,2:C,3:out}, meta
    """
    n_out = int(total * OUTLIER_FRAC);
    n_in = total - n_out
    n_a = int(n_in * pa);
    n_b = int(n_in * pb);
    n_c = n_in - n_a - n_b
    A0 = samp_yz_x(0.0, n_a, rng);
    B0 = samp_xz_y(0.0, n_b, rng);
    C0 = samp_xy_z(0.0, n_c, rng)
    A, nA = tilt_plane(A0, np.array([1.0, 0, 0]), RANDOM_ANGLE_DEG, rng)
    B, nB = tilt_plane(B0, np.array([0, 1.0, 0]), RANDOM_ANGLE_DEG, rng)
    C, nC = tilt_plane(C0, np.array([0, 0, 1.0]), RANDOM_ANGLE_DEG, rng)
    o = add_out(n_out, rng)
    pts = np.vstack([A, B, C, o])
    lbl = np.concatenate([np.full(n_a, 0), np.full(n_b, 1), np.full(n_c, 2), np.full(n_out, 3)])
    meta = {"truth_planes": [{"name": "A(≈x=0)", "normal": nA.tolist()},
                             {"name": "B(≈y=0)", "normal": nB.tolist()},
                             {"name": "C(≈z=0)", "normal": nC.tolist()}],
            "ratios": {"A": pa, "B": pb, "C": pc}}
    return pts, lbl, meta


# =========================
# RANSAC 平面
# =========================
@dataclass
class Plane:
    """
    功能: 平面模型 n·x + d = 0
    成员:
        normal (np.ndarray[3]): 单位法向
        d (float): 截距项
    """
    normal: np.ndarray
    d: float


def fit_plane_3(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Plane]:
    """
    功能: 三点拟合平面; 若近共线返回None
    输入:
        p1,p2,p3 (np.ndarray[3])
    输出:
        model (Plane|None)
    """
    v1, v2 = p2 - p1, p3 - p1
    n = np.cross(v1, v2);
    L = np.linalg.norm(n)
    if L < 1e-9: return None
    n = n / L;
    d = -float(n @ p1)
    # 统一符号: 使 d<=0，便于阅读
    if d > 0: n, d = -n, -d
    return Plane(n, d)


def p2p_dist(pts: np.ndarray, mdl: Plane) -> np.ndarray:
    """功能: 点到平面的有符号距离; 输入: pts(N,3), mdl; 输出: (N,)"""
    return pts @ mdl.normal + mdl.d


def ransac_plane(pts: np.ndarray, dist_th: float, iters: int, min_frac: float,
                 rng: np.random.Generator) -> Tuple[Optional[Plane], np.ndarray]:
    """
    功能: RANSAC估计单个平面; 局部掩码长度=当前点集
    输入:
        pts (np.ndarray[N,3])
        dist_th (float): 内点阈值(m)
        iters (int): 最大迭代数
        min_frac (float): 最小内点比例(相对N)
        rng (Generator): 随机源
    输出:
        model (Plane|None), in_mask (np.ndarray[bool,N])
    """
    N = pts.shape[0]
    if N < 3: return None, np.zeros(N, bool)
    best_m, best_mask, best_cnt = None, None, 0
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        m = fit_plane_3(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if m is None: continue
        d = np.abs(p2p_dist(pts, m))
        mask = d < dist_th;
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_m, best_mask, best_cnt = m, mask, cnt
    if best_m is None or best_cnt < max(int(min_frac * N), 3):
        return None, np.zeros(N, bool)
    return best_m, best_mask


def extract_planes(pts0: np.ndarray, dist_th: float, iters: int, min_frac: float,
                   stop_n: int, rng: np.random.Generator) -> List[Dict]:
    """
    功能: 迭代提取多个平面，返回“全局掩码/索引”便于可视化
    输入:
        pts0 (np.ndarray[N,3]): 初始点云
        dist_th, iters, min_frac, stop_n: RANSAC参数与停止阈值
        rng (Generator)
    输出:
        planes (List[dict]): 每项含:
            'normal':(3,), 'd':float, 'inlier_count':int, 'inlier_ratio':float,
            'mask_global':(N,), 'idx_global':(M,)
    """
    planes: List[Dict] = []
    N = pts0.shape[0]
    pts = pts0.copy()
    idx_global = np.arange(N, dtype=int)

    while pts.shape[0] > stop_n:
        m, mask_local = ransac_plane(pts, dist_th, iters, min_frac, rng)
        if m is None or mask_local.sum() < 3: break
        idx_in = idx_global[mask_local]
        mask_g = np.zeros(N, bool);
        mask_g[idx_in] = True
        planes.append({
            "normal": m.normal.copy(), "d": float(m.d),
            "inlier_count": int(mask_local.sum()),
            "inlier_ratio": float(mask_local.sum() / N),
            "mask_global": mask_g, "idx_global": idx_in.copy()
        })
        # remove inliers
        pts = pts[~mask_local];
        idx_global = idx_global[~mask_local]
    return planes


# =========================
# 可视化(简单实用)
# =========================
def make_plane_patch(n: np.ndarray, d: float, center: np.ndarray, size: float = 12.0) -> np.ndarray:
    """
    功能: 从法向/截距生成平面上的正方形补丁四点(用于绘制)
    输入:
        n (np.ndarray[3]): 单位法向
        d (float): 截距
        center (np.ndarray[3]): 平面上一点(如内点均值)
        size (float): 边长
    输出:
        quad (np.ndarray[4,3])
    """
    n = n / (np.linalg.norm(n) + 1e-12)
    tmp = np.array([1, 0, 0], float) if abs(n @ np.array([1, 0, 0])) < 0.9 else np.array([0, 1, 0], float)
    u = np.cross(n, tmp);
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    h = size * 0.5
    return np.vstack([center - h * u - h * v, center + h * u - h * v,
                      center + h * u + h * v, center - h * u + h * v])


def viz_mpl(pts: np.ndarray, planes: List[Dict], max_points: int = 12000) -> None:
    """
    功能: Matplotlib静态浏览(点+线框)
    输入:
        pts (np.ndarray[N,3]), planes (List[dict]), max_points (int)
    输出:
        无(弹窗)
    """
    if not HAS_MPL:
        print("Matplotlib 未安装，跳过。");
        return
    import numpy as np, matplotlib.pyplot as plt
    rngv = np.random.default_rng(0)
    if pts.shape[0] > max_points:
        idx = rngv.choice(pts.shape[0], max_points, replace=False);
        show = pts[idx]
    else:
        show = pts
    fig = plt.figure(figsize=(8, 6));
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(show[:, 0], show[:, 1], show[:, 2], s=1, alpha=0.5)
    for pl in planes:
        m = pl["mask_global"];
        ins = pts[m]
        if ins.size == 0: continue
        ctr = ins.mean(axis=0);
        quad = make_plane_patch(pl["normal"], pl["d"], ctr, 12.0)
        closed = np.vstack([quad, quad[0]]);
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], lw=2)
    ax.set(xlim=(0, CUBE_SIZE), ylim=(0, CUBE_SIZE), zlim=(0, CUBE_SIZE),
           xlabel="X(m)", ylabel="Y(m)", zlabel="Z(m)")
    ax.set_title("Point Cloud + Fitted Planes (wireframe)")
    plt.tight_layout();
    plt.show()


def viz_o3d(pts: np.ndarray, planes: List[Dict], sample_ratio: float = 0.5) -> None:
    """
    功能: Open3D交互浏览(点+平面补丁)
    输入:
        pts (np.ndarray[N,3]), planes (List[dict]), sample_ratio (0~1)
    输出:
        无(弹窗)
    """
    if not HAS_O3D:
        print("Open3D 未安装，跳过。");
        return
    import numpy as np, open3d as o3d
    N = pts.shape[0]
    if sample_ratio < 1.0:
        idx = np.random.default_rng(0).choice(N, max(1, int(N * sample_ratio)), replace=False)
        show = pts[idx]
    else:
        show = pts
    geoms = []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(show))
    pcd.paint_uniform_color([0.2, 0.2, 0.8]);
    geoms.append(pcd)
    for pl in planes:
        ins = pts[pl["mask_global"]];
        if ins.size == 0: continue
        ctr = ins.mean(axis=0);
        quad = make_plane_patch(pl["normal"], pl["d"], ctr, 12.0)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(quad)
        mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], int))
        mesh.compute_vertex_normals();
        mesh.paint_uniform_color([0.8, 0.8, 0.2]);
        geoms.append(mesh)
    o3d.visualization.draw_geometries(geoms)


# =========================
# 主流程
# =========================
def specs() -> List[Tuple[str, str, Tuple[float, ...]]]:
    """功能: 返回13组配置(name, kind, params)"""
    return [
        ("case01_single", "single", ()),
        ("case02_2planes_10_90", "two", (0.10,)),
        ("case03_2planes_20_80", "two", (0.20,)),
        ("case04_2planes_30_70", "two", (0.30,)),
        ("case05_2planes_40_60", "two", (0.40,)),
        ("case06_2planes_50_50", "two", (0.50,)),
        ("case07_3planes_30_35_35", "three", (0.30, 0.35, 0.35)),
        ("case08_3planes_40_30_30", "three", (0.40, 0.30, 0.30)),
        ("case09_3planes_50_25_25", "three", (0.50, 0.25, 0.25)),
        ("case10_3planes_60_20_20", "three", (0.60, 0.20, 0.20)),
        ("case11_3planes_70_15_15", "three", (0.70, 0.15, 0.15)),
        ("case12_3planes_80_10_10", "three", (0.80, 0.10, 0.10)),
        ("case13_3planes_90_05_05", "three", (0.90, 0.05, 0.05)),
    ]


def save_npz(path: Path, points: np.ndarray, labels: np.ndarray, meta: Dict) -> None:
    """功能: 保存压缩npz; 输入: 路径/点/标/元; 输出: 文件"""
    np.savez_compressed(path, points=points, labels=labels, meta=np.array([meta], dtype=object))


def main() -> None:
    """
    功能:
        生成13组数据(带法向扰动)，对每组做迭代RANSAC提取平面，写入npz与汇总CSV；
        最后一组弹出可视化(若装了对应库)。
    输入: 无
    输出: 文件到 OUT_DIR，控制台打印路径与示例。
    """
    rng = np.random.default_rng(RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    last_pts = None
    last_planes = None

    for name, kind, prm in specs():
        if kind == "single":
            pts, lbl, meta = ds_single(TOTAL_POINTS, rng)
        elif kind == "two":
            pts, lbl, meta = ds_two(prm[0], TOTAL_POINTS, rng)
        else:
            pts, lbl, meta = ds_three(*prm, TOTAL_POINTS, rng)

        meta.update({"name": name, "kind": kind, "params": prm,
                     "cube_size_m": CUBE_SIZE, "total_points": int(TOTAL_POINTS),
                     "noise_sigma_m": GAUSS_NOISE_SIGMA, "outlier_frac": OUTLIER_FRAC,
                     "random_angle_deg": RANDOM_ANGLE_DEG})
        save_npz(OUT_DIR / f"{name}.npz", pts, lbl, meta)

        planes = extract_planes(
            pts, RANSAC_DIST_THRESH, RANSAC_MAX_ITERS, RANSAC_MIN_INLIER_FRAC,
            STOP_THRESHOLD_NUMBER, rng
        )

        if planes:
            for i, pl in enumerate(planes, 1):
                rows.append({
                    "dataset": name, "plane_idx": i,
                    "normal_x": float(pl["normal"][0]),
                    "normal_y": float(pl["normal"][1]),
                    "normal_z": float(pl["normal"][2]),
                    "d": float(pl["d"]),
                    "inlier_count": int(pl["inlier_count"]),
                    "inlier_ratio": float(pl["inlier_ratio"]),
                    "total_points": int(pts.shape[0]),
                })
        else:
            rows.append({"dataset": name, "plane_idx": None,
                         "normal_x": np.nan, "normal_y": np.nan, "normal_z": np.nan,
                         "d": np.nan, "inlier_count": 0, "inlier_ratio": 0.0,
                         "total_points": int(pts.shape[0])})

        last_pts, last_planes = pts, planes

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ransac_summary.csv";
    df.to_csv(csv_path, index=False)
    print(f"输出目录: {OUT_DIR.resolve()}")
    print(f"汇总CSV: {csv_path.resolve()}")
    print("示例数据:", (OUT_DIR / "case06_2planes_50_50.npz").resolve())

    # 预览最后一组
    if last_pts is not None and last_planes is not None:
        if HAS_MPL:
            print("Matplotlib 预览 ...")
            viz_mpl(last_pts, last_planes, max_points=15000)
        if HAS_O3D:
            print("Open3D 预览 ...")
            viz_o3d(last_pts, last_planes, sample_ratio=0.5)


if __name__ == "__main__":
    main()

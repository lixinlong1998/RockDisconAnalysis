# -*- coding: utf-8 -*-
"""
点云平面生成与形变管线

功能简介:
    在 10×10×10 m 立方体内生成 1/2/3 个平面点云, 支持:
        - 规则网格采样(1 cm)
        - 高斯随机噪声(10%-50% of 1cm)
        - 系统性弯曲形变(0-1 弯曲度, 二次曲面)
        - 流形形变 I: 控制格点随机起伏 (Grid: 2x2/5x5/10x10, 双线性插值)
        - 流形形变 II: 正弦波浪起伏 (频率与Grid一致)
        - 2 面: 指定夹角(10..170°), 面积占比 (1:9..5:5)
        - 3 面: 以三棱锥近似(三侧面), 用【未经验证】映射将(三顶角和)映射为顶点高度,
               按占比缩放三面面积

实现思路:
    1) 几何构造层: 生成平面/三角面在立方体中的位置与方向(法向/切向基)
    2) 采样层: 依据 1 cm 网格在局部 (u,v) 参数域采样, 投影到 3D
    3) 形变层: 按顺序叠加 弯曲/格点插值/正弦波 与 高斯噪声
    4) 集合并裁剪到立方体范围
    5) 导出/可视化

输入变量:
    - cube_size: float, 立方体边长(默认 10.0)
    - plane_count: int, {1,2,3}, 平面数量
    - angle_deg: int 或 None,
        * 2 面时为两法向夹角
        * 3 面时解释为(三侧面顶角之和)的近似代理
    - area_ratio: tuple,
        * 2 面: (a,b) ∈ {(1,9),(2,8),(3,7),(4,6),(5,5)}
        * 3 面: (a,b,c) ∈ {题述的 8 组之一}
    - noise_level: float ∈ [0.1,0.5], 高斯噪声比例(相对 1 cm)
    - bend_kappa: float ∈ [0,1], 弯曲度
    - grid_n: int ∈ {2,5,10}, 用作两类“流形形变”的格点/频率
    - apply_grid_warp: bool, 是否应用“格点随机起伏”
    - apply_wave_warp: bool, 是否应用“正弦波浪起伏”

输出变量:
    - points: (N,3) float32 ndarray, 合并后的点云坐标
    - labels: (N,) int ndarray, 每点所属平面 ID (从 0 开始)
    - meta: dict, 记录生成参数与计时信息

依赖:
    numpy, scipy(可选, 若需要更高级插值这里未强制), matplotlib, logging

"""

import numpy as np
import time
import csv
import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import os

# ========================
# 常量与日志配置
# ========================
# CM = 0.01  # 1 cm   # 每个采样点间距 1 cm（即点云分辨率 0.01 m）
CM = 0.05
DEFAULT_EXPORT_DIR = r"E:\Database\_RockPoints\PlanesInCube"
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("plane_cloud_gen")


# ========================
# 计时装饰器
# ========================
def TimeIt(func):
    """装饰器: 记录函数耗时(秒)"""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        logger.info(f"[耗时] {func.__name__}: {(t1 - t0):.4f}s")
        return out

    return wrapper


# ========================
# 基础向量工具
# ========================
def Normalize(vec: np.ndarray) -> np.ndarray:
    """单位化向量"""
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return vec
    return vec / n


def OrthonormalBasis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由法向 n 构造正交基 (t1, t2, n)"""
    n = Normalize(n)
    # 选择与 n 不平行的向量
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, helper)
    t1 = Normalize(t1)
    t2 = np.cross(n, t1)
    t2 = Normalize(t2)
    return t1, t2, n


def RotateVectorAroundAxis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """罗德里格斯公式: 向量 v 绕单位轴 axis 旋转 angle_rad"""
    axis = Normalize(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1 - c)


# ========================
# 形变函数
# ========================
def BendDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float, kappa: float,
                     amp: float) -> np.ndarray:
    """
    系统性弯曲: 二次曲面法向位移
    输入:
        u,v: 网格坐标 (形状相同), 以补丁中心为原点的局部坐标
        u_half, v_half: 补丁半尺寸
        kappa: 弯曲度[0,1]
        amp: 基础幅值(米)
    输出:
        dn: 与 u,v 同形状, 标量法向位移
    """
    uu = (u / max(u_half, 1e-9)) ** 2
    vv = (v / max(v_half, 1e-9)) ** 2
    dn = kappa * amp * (uu + vv)
    return dn


def GridRandomDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float, grid_n: int, amp: float,
                           rng: np.random.Generator) -> np.ndarray:
    """
    流形形变 I: 控制格点随机起伏 (双线性插值)
    在 [-u_half, u_half]×[-v_half, v_half] 上生成 grid_n×grid_n 随机高度, 对 (u,v) 双线性插值。
    """
    # 生成控制点坐标与随机值
    u_coords = np.linspace(-u_half, u_half, grid_n)
    v_coords = np.linspace(-v_half, v_half, grid_n)
    ctrl = rng.uniform(-amp, amp, size=(grid_n, grid_n))

    # 将 (u,v) 映射到控制网格索引空间
    def interp_1d(x, x_coords):
        # 返回左索引 i 与插值因子 t, 使得 x ∈ [x_i, x_{i+1}]
        x = np.clip(x, x_coords[0], x_coords[-1])
        idx = np.searchsorted(x_coords, x) - 1
        idx = np.clip(idx, 0, len(x_coords) - 2)
        x0 = x_coords[idx]
        x1 = x_coords[idx + 1]
        t = np.where(np.abs(x1 - x0) < 1e-12, 0.0, (x - x0) / (x1 - x0))
        return idx, t

    iu, tu = interp_1d(u, u_coords)
    iv, tv = interp_1d(v, v_coords)

    # 双线性插值
    c00 = ctrl[iu, iv]
    c10 = ctrl[iu + 1, iv]
    c01 = ctrl[iu, iv + 1]
    c11 = ctrl[iu + 1, iv + 1]
    c0 = c00 * (1 - tu) + c10 * tu
    c1 = c01 * (1 - tu) + c11 * tu
    dn = c0 * (1 - tv) + c1 * tv
    return dn


def WaveDisplacement(u: np.ndarray, v: np.ndarray, u_half: float, v_half: float, grid_n: int, amp: float) -> np.ndarray:
    """
    流形形变 II: 正弦波浪起伏
    频率按 grid_n 与补丁尺度确定, 使 2/5/10 对应不同空间频率。
    """
    fu = grid_n / max(2 * u_half, 1e-9)
    fv = grid_n / max(2 * v_half, 1e-9)
    dn = amp * np.sin(2 * np.pi * fu * (u + u_half)) * np.sin(2 * np.pi * fv * (v + v_half))
    return dn


# ========================
# 采样与补丁构造
# ========================
@dataclass
class PatchSpec:
    center: np.ndarray  # (3,) 补丁中心
    normal: np.ndarray  # (3,) 法向
    size_u: float  # 沿 t1 的全尺寸 (米)
    size_v: float  # 沿 t2 的全尺寸 (米)
    is_triangle: bool  # 三角面(用于三棱锥侧面)或矩形面
    keep_halfspace: Optional[Tuple[np.ndarray, float]] = None
    # 可选: (n, d) 只保留满足 n·x + d >= 0 的半空间(用于裁剪/保证在立方体内, 这里通常不用)


class PlanePatchBuilder:
    """构造 1/2/3 面补丁的几何规格"""

    def __init__(self, cube_size: float = 10.0, rng: Optional[np.random.Generator] = None):
        self.cube_size = cube_size
        self.half = cube_size / 2.0
        self.center = np.array([self.half, self.half, self.half], dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng(1234)

    def _bounded_size(self, target_area: float, aspect: float = 1.0) -> Tuple[float, float]:
        """给定目标面积与宽高比, 计算在立方体允许范围内的 (size_u, size_v)"""
        size_u = np.sqrt(target_area * aspect)
        size_v = target_area / max(size_u, 1e-9)
        # 限制尺寸不超立方体
        size_u = float(np.clip(size_u, CM * 5, self.cube_size * 0.95))
        size_v = float(np.clip(size_v, CM * 5, self.cube_size * 0.95))
        return size_u, size_v

    def _make_rect_patch(self, normal: np.ndarray, area: float, center_offset: np.ndarray = None) -> PatchSpec:
        if center_offset is None:
            center = self.center.copy()
        else:
            center = self.center + center_offset
        size_u, size_v = self._bounded_size(area, aspect=1.0)
        return PatchSpec(center=center, normal=Normalize(normal), size_u=size_u, size_v=size_v, is_triangle=False)

    def _make_triangle_patch(self, normal: np.ndarray, edge_len: float, center_offset: np.ndarray = None) -> PatchSpec:
        """以近似的“等边三角面”为基础, 用 size_u/size_v 记录其外接矩形尺寸(用于参数化采样), 可视化时按三角形掩膜筛选"""
        if center_offset is None:
            center = self.center.copy()
        else:
            center = self.center + center_offset
        # 等边三角边长 edge_len -> 外接矩形近似
        height = np.sqrt(3) / 2 * edge_len
        size_u = edge_len
        size_v = height
        return PatchSpec(center=center, normal=Normalize(normal), size_u=size_u, size_v=size_v, is_triangle=True)

    # ----- 单面 -----
    def BuildOnePlane(self) -> List[PatchSpec]:
        # 取水平面 z ~ 常数, 法向朝 +z
        normal = np.array([0.0, 0.0, 1.0])
        area = (self.cube_size * 0.6) ** 2  # 让补丁占据中心区域(可调)
        return [self._make_rect_patch(normal, area, center_offset=np.array([0, 0, 0]))]

    # ----- 双面 -----
    def BuildTwoPlanes(self, angle_deg: int, area_ratio: Tuple[int, int]) -> List[PatchSpec]:
        a, b = area_ratio
        total_area = (self.cube_size * 0.8) ** 2
        area1 = total_area * (a / (a + b))
        area2 = total_area * (b / (a + b))

        # 第一面: 法向沿 +z
        n1 = np.array([0.0, 0.0, 1.0])
        # 第二面: 绕 x 轴旋转 angle 以形成与 n1 的夹角
        angle_rad = np.deg2rad(angle_deg)
        n2 = RotateVectorAroundAxis(n1, np.array([1.0, 0.0, 0.0]), angle_rad)

        # 轻微分离中心, 避免完全重叠
        off = self.cube_size * 0.05
        sgn = 1.0 if angle_deg <= 90 else -1.0
        p1 = self._make_rect_patch(n1, area1, center_offset=np.array([0, 0, sgn * off]))
        p2 = self._make_rect_patch(n2, area2, center_offset=np.array([0, 0, -sgn * off]))
        return [p1, p2]

    # ----- 三面(三棱锥侧面近似) -----
    def BuildThreePlanes(self, angle_sum_deg: int, area_ratio: Tuple[float, float, float]) -> List[PatchSpec]:
        # 【未经验证】角和 -> 顶点高度的经验映射
        h_min, h_max = 0.1, 4.0
        t = (180 - angle_sum_deg) / 170.0
        h = h_min + (h_max - h_min) * np.clip(t, 0.0, 1.0)

        # 基底取近似等边三角, 其中心在立方体中心下方/上方, 顶点在中心+z*h
        base_edge = self.cube_size * 0.9
        top = self.center + np.array([0.0, 0.0, h])
        base_z = self.center[2] - h * 0.6
        base_center = np.array([self.center[0], self.center[1], base_z])

        # 三侧面法向: 令三条从顶点指向基底三边中心的向量决定侧面方向
        # 先构造基底三角的三个顶点(在 z=base_z 平面)
        R = base_edge / np.sqrt(3) / 2 * 2  # 近似外接半径
        v0 = base_center + np.array([R, 0.0, 0.0])
        v1 = base_center + np.array([-R / 2, R * np.sqrt(3) / 2, 0.0])
        v2 = base_center + np.array([-R / 2, -R * np.sqrt(3) / 2, 0.0])
        base_pts = [v0, v1, v2]

        # 三边中点
        e01 = (v0 + v1) / 2
        e12 = (v1 + v2) / 2
        e20 = (v2 + v0) / 2
        edge_mids = [e01, e12, e20]

        normals = []
        for m in edge_mids:
            # 侧面大致法向 = 由顶点 top 指向边中点 m 的向量, 再与高度方向适配
            n = np.cross(m - top, np.array([0, 0, 1.0]))
            if np.linalg.norm(n) < 1e-9:
                n = m - top
            normals.append(Normalize(n))

        # 面积占比 -> 三角面的边长缩放
        a, b, c = area_ratio
        s = np.array([a, b, c], dtype=float)
        s = s / (s.max() + 1e-9)  # 以最大者为 1 缩放
        edge_scales = 0.5 + 0.5 * s  # 将 [0,1] 映射至 [0.5,1.0]，避免过小

        # 构造三侧面(等边三角近似), 用 edge_len 控制面积
        edge_len_base = self.cube_size * 0.9
        patches = []
        for i in range(3):
            n = normals[i]
            edge_len = float(edge_len_base * edge_scales[i])
            # 把三角面的“中心”放在顶点与边中点的中线附近, 使三面共享近似顶点区域
            center_offset = (edge_mids[i] + top) / 2 - self.center
            patches.append(self._make_triangle_patch(n, edge_len, center_offset=center_offset))
        return patches


# ========================
# 点云生成器
# ========================
class PlanePointCloudGenerator:
    """按 PatchSpec 采样点云并叠加形变与噪声"""

    def __init__(self, cube_size: float = 10.0, seed: int = 2025):
        self.cube_size = cube_size
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _triangle_mask(u: np.ndarray, v: np.ndarray, size_u: float, size_v: float) -> np.ndarray:
        """
        在外接矩形内生成等边三角的二值掩膜:
        顶边平行 t1, 高度 size_v, 底边在 v = -size_v/2, 顶点在 v = +size_v/2
        """
        u_half = size_u / 2.0
        v_half = size_v / 2.0
        # 三角形边界: |u| <= ( (v + v_half) / size_v ) * u_half * 2
        # 当 v = -v_half 时, 右侧=0; 当 v = +v_half 时, 右侧 = u_half*2
        coef = np.clip((v + v_half) / (2 * v_half + 1e-9), 0.0, 1.0)
        u_limit = coef * (u_half * 2)
        mask = (np.abs(u) <= u_limit) & (v >= -v_half) & (v <= v_half)
        return mask

    @TimeIt
    def SamplePatch(self,
                    patch: PatchSpec,
                    step: float = CM,
                    bend_kappa: float = 0.0,
                    grid_n: Optional[int] = None,
                    bend_amp: float = 0.2,
                    grid_amp: float = 0.05,
                    wave_amp: float = 0.05,
                    apply_grid_warp: bool = False,
                    apply_wave_warp: bool = False,
                    noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样单个补丁并施加形变与噪声
        返回:
            pts: (M,3)
            lbl: (M,) 全为同一平面ID占位, 由外层赋值
        """
        t1, t2, n = OrthonormalBasis(patch.normal)

        # 局部网格 (u,v)
        u_half = patch.size_u / 2.0
        v_half = patch.size_v / 2.0
        u_vals = np.arange(-u_half, u_half + 1e-9, step)
        v_vals = np.arange(-v_half, v_half + 1e-9, step)
        U, V = np.meshgrid(u_vals, v_vals, indexing='xy')

        if patch.is_triangle:
            mask = self._triangle_mask(U, V, patch.size_u, patch.size_v)
            U = U[mask]
            V = V[mask]

        # ------- 这里是修复的核心 -------
        base = patch.center
        if U.ndim == 2:
            # 矩形面: (m,n,3) -> (mn,3)
            pts = (base[None, None, :]
                   + U[..., None] * t1[None, None, :]
                   + V[..., None] * t2[None, None, :]).reshape(-1, 3)
            # 后续形变/噪声计算需要一一对应，展平 U,V
            U = U.reshape(-1)
            V = V.reshape(-1)
        else:
            # 三角面: (M,3)
            pts = (base[None, :]
                   + U[:, None] * t1[None, :]
                   + V[:, None] * t2[None, :])
        # --------------------------------

        # 弯曲形变(法向位移)
        if bend_kappa > 0.0:
            dn = BendDisplacement(U, V, u_half, v_half, bend_kappa, amp=bend_amp)
            pts = pts + dn[:, None] * n[None, :]

        # 格点随机起伏
        if apply_grid_warp and (grid_n is not None):
            dn = GridRandomDisplacement(U, V, u_half, v_half, grid_n, amp=grid_amp, rng=self.rng)
            pts = pts + dn[:, None] * n[None, :]

        # 正弦波浪
        if apply_wave_warp and (grid_n is not None):
            dn = WaveDisplacement(U, V, u_half, v_half, grid_n, amp=wave_amp)
            pts = pts + dn[:, None] * n[None, :]

        # 高斯噪声
        if noise_level > 0.0:
            sigma = noise_level * CM
            noise = self.rng.normal(0.0, sigma, size=pts.shape)
            pts = pts + noise

        # 裁剪到立方体
        pts = np.clip(pts, 0.0, self.cube_size)

        labels = np.zeros((pts.shape[0],), dtype=int)
        return pts.astype(np.float32), labels

    @TimeIt
    def Build(self,
              plane_count: int,
              angle_deg: Optional[int] = None,
              area_ratio: Optional[Tuple[float, ...]] = None,
              step: float = CM,
              bend_kappa: float = 0.0,
              bend_amp: float = 0.2,
              grid_amp: float = 0.05,
              wave_amp: float = 0.05,
              grid_n: Optional[int] = None,
              apply_grid_warp: bool = False,
              apply_wave_warp: bool = False,
              noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        生成完整点云
        """
        # 1) 构造补丁规格
        builder = PlanePatchBuilder(self.cube_size, rng=self.rng)
        if plane_count == 1:
            patches = builder.BuildOnePlane()
        elif plane_count == 2:
            assert angle_deg is not None and area_ratio is not None and len(area_ratio) == 2
            patches = builder.BuildTwoPlanes(angle_deg, (int(area_ratio[0]), int(area_ratio[1])))
        elif plane_count == 3:
            assert angle_deg is not None and area_ratio is not None and len(area_ratio) == 3
            patches = builder.BuildThreePlanes(int(angle_deg), area_ratio)  # 三顶角和近似
        else:
            raise ValueError("plane_count 只能是 1/2/3")

        # 2) 逐补丁采样+形变
        all_pts = []
        all_lbl = []
        for i, p in enumerate(patches):
            pts, lbl = self.SamplePatch(
                p, step=step, bend_kappa=bend_kappa,
                grid_n=grid_n, bend_amp=bend_amp, grid_amp=grid_amp, wave_amp=wave_amp,
                apply_grid_warp=apply_grid_warp,
                apply_wave_warp=apply_wave_warp,
                noise_level=noise_level
            )
            lbl[:] = i
            all_pts.append(pts)
            all_lbl.append(lbl)

        points = np.vstack(all_pts)
        labels = np.concatenate(all_lbl)

        meta = dict(
            plane_count=plane_count,
            angle_deg=angle_deg,
            area_ratio=area_ratio,
            step=step,
            bend_kappa=bend_kappa,
            grid_n=grid_n,
            apply_grid_warp=apply_grid_warp,
            apply_wave_warp=apply_wave_warp,
            noise_level=noise_level,
            n_points=int(points.shape[0])
        )
        logger.info(f"[完成] 生成点数: {points.shape[0]}")
        return points, labels, meta


# ========================
# 导出与可视化
# ========================
def SaveAsXYZ(path: str, pts: np.ndarray, lbl: Optional[np.ndarray] = None):
    """保存为 .xyz (可选第4列为标签)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if lbl is not None:
        arr = np.hstack([pts, lbl[:, None].astype(float)])
    else:
        arr = pts
    np.savetxt(path, arr, fmt="%.6f")
    logger.info(f"[导出] XYZ: {path}")


def SaveAsPLY(path: str, pts: np.ndarray, lbl: Optional[np.ndarray] = None):
    """保存为 PLY (ASCII)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = pts.shape[0]
    has_label = lbl is not None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_label:
            f.write("property uchar label\n")
        f.write("end_header\n")
        if has_label:
            for i in range(n):
                f.write(f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} {int(lbl[i])}\n")
        else:
            for i in range(n):
                f.write(f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f}\n")
    logger.info(f"[导出] PLY: {path}")


def QuickShow3D(pts: np.ndarray, lbl: Optional[np.ndarray] = None, max_show: int = 200000):
    """简单 3D 可视化(采样显示), 适合快速检查"""
    import matplotlib.pyplot as plt
    n = pts.shape[0]
    idx = np.arange(n)
    if n > max_show:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_show, replace=False)
    sub = pts[idx]
    if lbl is not None:
        sub_lbl = lbl[idx]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    if lbl is None:
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.2)
    else:
        # 分平面着色
        for lab in np.unique(sub_lbl):
            mask = (sub_lbl == lab)
            ax.scatter(sub[mask, 0], sub[mask, 1], sub[mask, 2], s=0.2, label=f"P{lab}")
        ax.legend(loc="upper right", markerscale=5)
    ax.set_xlabel("X (m)");
    ax.set_ylabel("Y (m)");
    ax.set_zlabel("Z (m)")
    ax.set_xlim(0, 10);
    ax.set_ylim(0, 10);
    ax.set_zlim(0, 10)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


# ========================
# 示例主函数 (可直接改参数批量生成)
# ========================
if __name__ == "__main__":
    gen = PlanePointCloudGenerator(cube_size=10.0, seed=42)
    # ========================
    # 批量生成：plane_count = 2
    # 命名: Plane2_Ang{ang}_Ara{a}-{b}_Gno{gno}_Ben{ben}_Grid{grid}_Sin{sinp}.ply
    # ========================
    bend_amp_base = 0.2
    grid_amp_base = 0.05
    wave_amp_base = 0.05

    angle_list = [20, 40, 60, 80, 100, 120, 140, 160]  # 两平面夹角(度)
    area_ratio_list = [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]  # 面积占比
    noise_percent_list = [0, 20, 40, 60, 80, 100]  # 高斯噪声 %
    bend_percent_list = [0, 20, 40, 60, 80, 100]  # 弯曲度 %
    grid_n_list = [2, 4, 6, 8, 10]  # Grid: 2x2,4x4,...
    wave_percent_list = [0, 20, 40, 60, 80, 100]  # 正弦起伏 %

    batch_dir2 = os.path.join(DEFAULT_EXPORT_DIR, "batch_plane2")
    os.makedirs(batch_dir2, exist_ok=True)
    manifest2_path = os.path.join(batch_dir2, "manifest_plane2.csv")

    # 写表头(若不存在)
    write_header2 = not os.path.exists(manifest2_path)
    with open(manifest2_path, "a", newline="", encoding="utf-8") as f2:
        writer2 = csv.writer(f2)
        if write_header2:
            writer2.writerow([
                "filename",
                "plane_count",
                "angle_deg",
                "area_ratio",
                "noise_percent",
                "bend_percent",
                "grid_n",
                "wave_percent",
                "bend_amp",
                "grid_amp",
                "wave_amp_abs",
                "n_points",
                # 两个真值平面方程 (ax+by+cz+d=0)
                "A1", "B1", "C1", "D1",
                "A2", "B2", "C2", "D2",
            ])

        total_count2 = 0
        skipped2 = 0

        for ang in angle_list:
            for (a_part, b_part) in area_ratio_list:
                ar_str = f"{a_part}:{b_part}"
                for gno in noise_percent_list:
                    noise_level = gno / 100.0
                    for ben in bend_percent_list:
                        bend_kappa = ben / 100.0
                        for grid_n in grid_n_list:
                            for sinp in wave_percent_list:
                                wave_amp = (sinp / 100.0) * wave_amp_base
                                apply_wave = (sinp > 0)

                                fname = f"Plane2_Ang{ang}_Ara{a_part}-{b_part}_Gno{gno}_Ben{ben}_Grid{grid_n}_Sin{sinp}.ply"
                                fpath = os.path.join(batch_dir2, fname)

                                if os.path.exists(fpath):
                                    skipped2 += 1
                                    continue

                                # 构建两平面数据
                                pts, lbl, meta = gen.Build(
                                    plane_count=2,
                                    angle_deg=ang,
                                    area_ratio=(a_part, b_part),
                                    step=CM,
                                    bend_kappa=bend_kappa,
                                    bend_amp=bend_amp_base,
                                    grid_amp=grid_amp_base,
                                    wave_amp=wave_amp,
                                    grid_n=grid_n,
                                    apply_grid_warp=True,
                                    apply_wave_warp=apply_wave,
                                    noise_level=noise_level
                                )

                                # 保存
                                SaveAsPLY(fpath, pts, lbl)

                                # 读取两条真值平面方程
                                if "gt_planes" in meta and len(meta["gt_planes"]) >= 2:
                                    A1 = meta["gt_planes"][0]["a"];
                                    B1 = meta["gt_planes"][0]["b"];
                                    C1 = meta["gt_planes"][0]["c"];
                                    D1 = meta["gt_planes"][0]["d"]
                                    A2 = meta["gt_planes"][1]["a"];
                                    B2 = meta["gt_planes"][1]["b"];
                                    C2 = meta["gt_planes"][1]["c"];
                                    D2 = meta["gt_planes"][1]["d"]
                                else:
                                    A1 = B1 = C1 = D1 = A2 = B2 = C2 = D2 = (float("nan"))

                                # 写清单
                                writer2.writerow([
                                    fname,
                                    2,  # plane_count
                                    ang,  # angle_deg
                                    ar_str,  # area_ratio "a:b"
                                    gno,  # noise_percent
                                    ben,  # bend_percent
                                    grid_n,  # grid_n
                                    sinp,  # wave_percent
                                    bend_amp_base,
                                    grid_amp_base,
                                    wave_amp,  # 绝对幅值(米)
                                    meta.get("n_points", pts.shape[0]),
                                    A1, B1, C1, D1,
                                    A2, B2, C2, D2
                                ])

                                total_count2 += 1

        logger.info(
            f"[批量完成: plane2] 新生成: {total_count2} 个文件, 跳过(已存在): {skipped2} 个。清单: {manifest2_path}")

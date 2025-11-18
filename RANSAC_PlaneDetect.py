# -*- coding: utf-8 -*-
import os, re, time, csv, sys
import numpy as np

CM = 0.01  # 1 cm

# 尝试导入 open3d 或 pyransac3d
_BACKEND = None
try:
    import open3d as o3d

    _BACKEND = "open3d"
except Exception:
    try:
        import pyransac3d as pyrsc

        _BACKEND = "pyransac3d"
    except Exception:
        _BACKEND = "numpy"

print(f"[INFO] RANSAC 后端: {_BACKEND}")


# ---------- 基础 IO ----------
def load_points(path):
    ext = os.path.splitext(path)[1].lower()
    if _BACKEND == "open3d":
        if ext == ".ply":
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points, dtype=np.float64)
            return pts
        elif ext == ".xyz":
            return np.loadtxt(path, dtype=np.float64)[:, :3]
    else:
        # 简易 ASCII PLY 加载（配合你导出的格式）
        if ext == ".ply":
            pts = []
            with open(path, "r", encoding="utf-8") as f:
                header = True
                for line in f:
                    line = line.strip()
                    if header:
                        if line == "end_header":
                            header = False
                        continue
                    if not line:
                        continue
                    parts = line.split()
                    # 可能有 label 列
                    x, y, z = map(float, parts[:3])
                    pts.append((x, y, z))
            return np.asarray(pts, dtype=np.float64)
        elif ext == ".xyz":
            return np.loadtxt(path, dtype=np.float64)[:, :3]
    raise ValueError(f"不支持的文件类型或后端无法读取: {path}")


def save_colored_cloud(out_path, pts, inlier_mask):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if _BACKEND == "open3d":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        colors = np.zeros_like(pts)
        colors[inlier_mask] = [1.0, 0, 0]  # inliers 白色
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    else:
        # 写 ASCII PLY (x y z r g b)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(pts.shape[0]):
                r, g, b = (0, 255, 0) if inlier_mask[i] else (255, 0, 0)
                f.write(f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} {r} {g} {b}\n")


# ---------- 阈值策略 ----------
_fname_gno_re = re.compile(r"Gno(\d+)", re.IGNORECASE)


def parse_noise_sigma_from_name(fname):
    """
    从文件名中解析 GnoXX -> sigma = (XX/100)*CM
    若未解析到，返回稳健默认：sigma = 0.3*CM
    """
    m = _fname_gno_re.search(fname)
    if m:
        gno = int(m.group(1))
        return (gno / 100.0) * CM
    return 0.3 * CM


def pick_threshold(fname):
    sigma = parse_noise_sigma_from_name(os.path.basename(fname))
    thr = max(2 * CM, 3.0 * sigma)  # 稳健阈值：至少 2cm，或 3σ
    return float(thr)


# ---------- RANSAC 实现 ----------
def ransac_plane_open3d(pts, threshold):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # ransac_n=3 即最小三点定平面；迭代次数可按需提高
    model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                       ransac_n=3,
                                       num_iterations=2000)
    # open3d 模型: a,b,c,d (ax+by+cz+d=0)
    inliers = np.array(inliers, dtype=int)
    mask = np.zeros((pts.shape[0],), dtype=bool)
    mask[inliers] = True
    return model, mask


def ransac_plane_pyransac3d(pts, threshold):
    plane = pyrsc.Plane()
    # 返回平面参数 (a,b,c,d)、内点索引
    A, inliers = plane.fit(pts, thresh=threshold, maxIteration=2000)
    inliers = np.array(inliers, dtype=int)
    mask = np.zeros((pts.shape[0],), dtype=bool)
    mask[inliers] = True
    return A, mask


def ransac_plane_numpy(pts, threshold):
    """
    简易 NumPy 版本（仅用于兜底）：
    - 随机采样三点拟合平面
    - 评估内点数量
    - 循环迭代
    """
    rng = np.random.default_rng(0)
    n = pts.shape[0]
    best_mask = None
    best_cnt = -1
    best_plane = None

    def fit_plane(p0, p1, p2):
        # 通过三点求平面 ax+by+cz+d=0
        v1 = p1 - p0
        v2 = p2 - p0
        nrm = np.cross(v1, v2)
        if np.linalg.norm(nrm) < 1e-12:
            return None
        a, b, c = nrm / np.linalg.norm(nrm)
        d = -np.dot([a, b, c], p0)
        return np.array([a, b, c, d], dtype=np.float64)

    def dist_to_plane(plane, P):
        a, b, c, d = plane
        return np.abs(P @ np.array([a, b, c]) + d) / (np.sqrt(a * a + b * b + c * c) + 1e-12)

    iters = 2000
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        plane = fit_plane(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if plane is None:
            continue
        d = dist_to_plane(plane, pts)
        mask = d <= threshold
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_mask = mask
            best_plane = plane
    if best_mask is None:
        # 退化兜底
        best_plane = np.array([0, 0, 1, -np.median(pts[:, 2])], dtype=np.float64)
        d = np.abs(pts[:, 2] - np.median(pts[:, 2]))
        best_mask = d <= threshold
    return best_plane, best_mask


def ransac_plane(pts, threshold):
    if _BACKEND == "open3d":
        return ransac_plane_open3d(pts, threshold)
    elif _BACKEND == "pyransac3d":
        return ransac_plane_pyransac3d(pts, threshold)
    else:
        return ransac_plane_numpy(pts, threshold)


# ---------- 批量处理 ----------
def process_one(path, out_dir_color):
    t0 = time.perf_counter()
    pts = load_points(path)
    thr = pick_threshold(path)
    model, mask = ransac_plane(pts, thr)
    t1 = time.perf_counter()

    # 保存着色点云
    color_name = os.path.splitext(os.path.basename(path))[0] + "_colored.ply"
    save_colored_cloud(os.path.join(out_dir_color, color_name), pts, mask)

    res = {
        "file": os.path.basename(path),
        "backend": _BACKEND,
        "n_points": int(pts.shape[0]),
        "threshold_m": f"{thr:.6f}",
        "inliers": int(mask.sum()),
        "inlier_ratio": f"{(mask.sum() / max(1, pts.shape[0])):.6f}",
        "plane_a": f"{model[0]:.10f}",
        "plane_b": f"{model[1]:.10f}",
        "plane_c": f"{model[2]:.10f}",
        "plane_d": f"{model[3]:.10f}",
        "runtime_ms": f"{(t1 - t0) * 1000:.2f}",
    }
    return res


def run_batch(input_dir, results_csv="ransac_results.csv"):
    assert os.path.isdir(input_dir), f"目录不存在: {input_dir}"
    out_dir_color = os.path.join(input_dir, "colored")
    os.makedirs(out_dir_color, exist_ok=True)

    csv_path = os.path.join(input_dir, results_csv)
    write_header = not os.path.exists(csv_path)
    cnt = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "file", "backend", "n_points", "threshold_m",
                "inliers", "inlier_ratio",
                "plane_a", "plane_b", "plane_c", "plane_d",
                "runtime_ms"
            ])
        files = [x for x in os.listdir(input_dir) if x.lower().endswith((".ply", ".xyz"))]
        files.sort()
        for name in files:
            path = os.path.join(input_dir, name)
            try:
                res = process_one(path, out_dir_color)
                writer.writerow([
                    res["file"], res["backend"], res["n_points"], res["threshold_m"],
                    res["inliers"], res["inlier_ratio"],
                    res["plane_a"], res["plane_b"], res["plane_c"], res["plane_d"],
                    res["runtime_ms"]
                ])
                cnt += 1
                if cnt % 50 == 0:
                    print(f"[INFO] 进度: {cnt}/{len(files)} ...")
            except Exception as e:
                print(f"[WARN] 处理失败: {name} -> {e}")

    print(f"[DONE] 处理完成: {cnt} / {len(files)} 个文件。结果: {csv_path}; 着色点云目录: {out_dir_color}")


# ---------- 入口 ----------
if __name__ == "__main__":
    # 修改为你的批量目录；你上条消息里是：
    # E:\Database\_RockPoints\PlanesInCube\batch_plane1
    INPUT_DIR = r"E:\Database\_RockPoints\PlanesInCube\batch_plane1"
    run_batch(INPUT_DIR)

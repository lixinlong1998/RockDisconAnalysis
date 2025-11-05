# -*- coding: utf-8 -*-
"""
RANSAC 结果图表分析：不同噪声类型与强度的影响
依赖: numpy, pandas, matplotlib
    conda/pip 安装: pandas, matplotlib
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- 配置 ---------
ROOT = r"E:\Database\_RockPoints\PlanesInCube\batch_plane1"
RES_CSV = os.path.join(ROOT, "ransac_results.csv")
MANIFEST_CSV = os.path.join(ROOT, "manifest_plane1.csv")  # 可选
OUTDIR = os.path.join(ROOT, "analysis_charts")
os.makedirs(OUTDIR, exist_ok=True)

# --------- 解析工具：从文件名抽参数 ---------
re_gno = re.compile(r"Gno(\d+)", re.IGNORECASE)
re_grid = re.compile(r"Grid(\d+)", re.IGNORECASE)
re_sin = re.compile(r"Sin(\d+)", re.IGNORECASE)
re_ben = re.compile(r"Ben(\d+)", re.IGNORECASE)


def parse_params_from_name(name: str):
    gno = re_gno.search(name)
    grid = re_grid.search(name)
    sinp = re_sin.search(name)
    ben = re_ben.search(name)
    return {
        "noise_percent": int(gno.group(1)) if gno else None,
        "grid_n": int(grid.group(1)) if grid else None,
        "wave_percent": int(sinp.group(1)) if sinp else None,
        "bend_percent": int(ben.group(1)) if ben else None,
    }


# --------- 读数据并合并 ---------
# ransac_results.csv: 由之前脚本输出，含 file, inlier_ratio 等
res = pd.read_csv(RES_CSV)

# 兼容 inlier_ratio 类型
if "inlier_ratio" in res.columns:
    # 有些记录可能是字符串
    res["inlier_ratio"] = pd.to_numeric(res["inlier_ratio"], errors="coerce")
else:
    raise RuntimeError("缺少列 inlier_ratio，请确认 ransac_results.csv。")

# 从文件名解析参数（作为兜底/校验）
parsed = res["file"].apply(parse_params_from_name).apply(pd.Series)
for c in ["noise_percent", "grid_n", "wave_percent", "bend_percent"]:
    if c not in res.columns:
        res[c] = parsed[c]
    else:
        # 若已有列但存在缺失，用解析结果补齐
        res[c] = res[c].fillna(parsed[c])

# 如果有 manifest，可进一步校验/覆盖
if os.path.exists(MANIFEST_CSV):
    man = pd.read_csv(MANIFEST_CSV)
    # 只保留 manifest 中关心的列，避免命名冲突
    cols = ["filename", "noise_percent", "grid_n", "wave_percent", "bend_percent", "n_points"]
    has = [c for c in cols if c in man.columns]
    man2 = man[has].rename(columns={"filename": "file"})
    res = pd.merge(res, man2, on="file", how="left", suffixes=("", "_m"))
    # 若存在 *_m 列，优先用 manifest
    for key in ["noise_percent", "grid_n", "wave_percent", "bend_percent", "n_points"]:
        if key + "_m" in res.columns:
            res[key] = res[key].fillna(res[key + "_m"])
    # 清理多余列
    res = res.drop(columns=[c for c in res.columns if c.endswith("_m")], errors="ignore")

# 规范数据类型/去坏值
for c in ["noise_percent", "grid_n", "wave_percent", "bend_percent", "n_points"]:
    if c in res.columns:
        res[c] = pd.to_numeric(res[c], errors="coerce")
res = res.dropna(subset=["inlier_ratio", "noise_percent", "grid_n", "wave_percent", "bend_percent"])

# --------- 图1：内点率 vs 高斯噪声（全局） ---------
g1 = (res.groupby("noise_percent")["inlier_ratio"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .sort_values("noise_percent"))
x = g1["noise_percent"].values
y = g1["mean"].values
yerr = g1["std"].values

plt.figure(figsize=(7, 5))
plt.plot(x, y, marker="o")
# 阴影带：±1 std（可按需改为 SEM）
y1 = y - yerr
y2 = y + yerr
plt.fill_between(x, y1, y2, alpha=0.2)
plt.xlabel("Gaussian noise level (%)")
plt.ylabel("Inlier ratio")
plt.title("RANSAC robustness vs Gaussian noise (mean ± std)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig1_inlier_vs_gaussian.png"), dpi=150)
plt.close()

# --------- 图2：不同 Grid 密度下，内点率 vs 正弦起伏 ---------
plt.figure(figsize=(7, 5))
for gn in sorted(res["grid_n"].dropna().unique()):
    sub = (res[res["grid_n"] == gn]
           .groupby("wave_percent")["inlier_ratio"]
           .agg(["mean", "count"])
           .reset_index()
           .sort_values("wave_percent"))
    if sub.empty:
        continue
    plt.plot(sub["wave_percent"].values, sub["mean"].values, marker="o", label=f"Grid {int(gn)}x{int(gn)}")
plt.xlabel("Sine warp amplitude (%)")
plt.ylabel("Inlier ratio")
plt.title("Effect of sine warp under different grid densities")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig2_inlier_vs_sine_by_grid.png"), dpi=150)
plt.close()

# --------- 图3：热力图（Grid × 正弦起伏） ---------
pivot = (res.groupby(["grid_n", "wave_percent"])["inlier_ratio"]
         .mean()
         .reset_index()
         .pivot(index="wave_percent", columns="grid_n", values="inlier_ratio")
         .sort_index())
plt.figure(figsize=(7, 5))
# 使用 imshow 展示，插值关闭，坐标刻度取实际取值
im = plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.colorbar(im, label="Mean inlier ratio")
plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns.astype(int))
plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index.astype(int))
plt.xlabel("Grid N (N×N)")
plt.ylabel("Sine warp amplitude (%)")
plt.title("Inlier ratio heatmap: Grid density × Sine amplitude")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig3_heatmap_grid_vs_sine.png"), dpi=150)
plt.close()

# --------- 图4：固定高斯噪声分位点(低/中/高)，内点率 vs 弯曲度 ---------
# 选三个代表：低(25分位)、中(50)、高(75)
qs = res["noise_percent"].quantile([0.25, 0.5, 0.75]).values.astype(int)
plt.figure(figsize=(7, 5))
for q in qs:
    # 允许 ±10% 的带宽以聚合足量样本
    band = res[(res["noise_percent"] >= q - 10) & (res["noise_percent"] <= q + 10)]
    if band.empty:
        continue
    sub = (band.groupby("bend_percent")["inlier_ratio"]
           .agg(["mean", "count"])
           .reset_index()
           .sort_values("bend_percent"))
    plt.plot(sub["bend_percent"].values, sub["mean"].values, marker="o", label=f"Gno≈{q}%")
plt.xlabel("Bend curvature (%)")
plt.ylabel("Inlier ratio")
plt.title("Effect of curvature under low/medium/high Gaussian noise")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig4_inlier_vs_bend_by_gno.png"), dpi=150)
plt.close()

print(f"[DONE] 图表已输出到: {OUTDIR}")

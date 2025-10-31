#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMS-based discontinuity grouping + DBSCAN splitting + plane fitting (A,B,C,D) + curvature output

Input : PLY point cloud (vertices only)
Output: TXT (TSV) with columns
        X\tY\tZ\tcluster_id\tsubcluster_id\tfacet_id\tA\tB\tC\tD\tcurvature

Notes
-----
1) "cluster_id"   = group id from mean-shift on the unit sphere of normals (after denoising & merging).
2) "subcluster_id"= DBSCAN component id within each group (>=0). Noise inside group is -1.
3) "facet_id"     = global unique id of each (group, subcluster) facet; -1 for unassigned points.
4) Removed-by-denoising points retain curvature but have cluster_id/subcluster_id/facet_id = -1 and A..D = NaN.

Dependencies: numpy, scipy, scikit-learn, plyfile
Install:     pip install numpy scipy scikit-learn plyfile

Author: ChatGPT (GPT-5 Thinking)


python _MMSDiscontinuityPipeline.py `
-i "E:\Database\_RockPoints\TSDK_Rockfall_RegularClip\TSDK_Rockfall_13_P1_ORG.ply" `
-o "D:\Research\20250313_RockFractureSeg\Code\qfacet_gpu\data\facet_export\TSDK_Rockfall_13_P1_ORG_MMS.txt" `
--gmax 0.06 `
--ms-mode landmark --ms-sample 0.12 --ms-seed-max 40000 --ms-iter 25 `
--R 10 --Ts 1.0 --Tm 5 --Tf 10 --C1 0.01 --C2 10 --eps 0.1
"""

from __future__ import annotations
import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


# ------------------------------
# Utilities
# ------------------------------

def setup_logger(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def read_ply_vertices(path: str) -> np.ndarray:
    """Read PLY and return Nx3 float64 array of vertices.
    Accepts binary or ascii PLY files with 'vertex' element and properties x,y,z.
    """
    ply = PlyData.read(path)
    if 'vertex' not in ply:
        raise ValueError("No 'vertex' element in PLY")
    v = ply['vertex']
    if not all(p in v.data.dtype.names for p in ('x', 'y', 'z')):
        raise ValueError("PLY 'vertex' element must have x,y,z")
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float64, copy=False)
    return xyz


def write_txt_tsv(path: str, header: List[str], data: np.ndarray):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\t".join(header) + "\n")
        np.savetxt(f, data, fmt='%.9f\t%.9f\t%.9f\t%d\t%d\t%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f')


def normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Return angle (radians) between two 3D unit vectors."""
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.acos(dot)


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def angular_radius_to_chordal_radius(theta_rad: float) -> float:
    """Convert angular radius on the unit sphere to Euclidean chordal radius in R^3.
    r_chord = 2 * sin(theta/2)
    """
    return 2.0 * math.sin(max(theta_rad, 0.0) * 0.5)


# ------------------------------
# PCA normals + curvature
# ------------------------------

def compute_knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) neighbor indices for each point (including self), and k is the knn"""
    tree = cKDTree(points)
    # query returns distances and indices; include self
    dists, idxs = tree.query(points, k=k)
    if k == 1:
        idxs = idxs[:, None]
    return idxs


def compute_normals_and_curvature(points: np.ndarray, knn_indices: np.ndarray, batch: int = 50000,
                                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-point PCA normal (smallest eigenvector) and curvature = l_min / (l1+l2+l3).
    Uses batched vectorized covariance + batched eigh.
    Returns:
        normals: (N,3) float32 unit vectors
        curvature: (N,) float32
    """
    N, k = knn_indices.shape
    normals = np.zeros((N, 3), dtype=np.float32)
    curvature = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for s in range(0, N, batch):
        e = min(s + batch, N)
        idx_chunk = knn_indices[s:e]  # (B, k)
        # Gather neighbor coordinates -> (B, k, 3)
        neigh = points[idx_chunk]
        # Center
        mu = neigh.mean(axis=1, keepdims=True)  # (B,1,3)
        X = neigh - mu  # (B,k,3)
        # Covariance: (B,3,3) = X^T X / k
        cov = np.einsum('bki,bkj->bij', X, X) / float(k)
        # Batched eigen-decomp (ascending eigenvalues)
        w, V = np.linalg.eigh(cov)  # w: (B,3), V: (B,3,3)
        # Smallest eigenvector is normal
        n = V[:, :, 0]
        # Normalize just in case
        n = normalize_rows(n.astype(np.float64)).astype(np.float32)
        normals[s:e] = n
        # Curvature
        wsum = np.maximum(w.sum(axis=1), 1e-15)
        curvature[s:e] = (w[:, 0] / wsum).astype(np.float32)

        if verbose and (s == 0 or (s // batch) % 10 == 0 or e == N):
            logging.debug(f"Normals batch {s}:{e} / {N}")

    logging.info(f"Computed normals+curvature for {N} points in {time.time() - t0:.2f}s")
    return normals, curvature


def regulate_normals(normals: np.ndarray, sensor_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """Flip normals so that dot(n, sensor_vec) >= 0.
    If sensor_vec is None, use +Z axis.
    """
    if sensor_vec is None:
        sensor_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    sv = sensor_vec / (np.linalg.norm(sensor_vec) + 1e-12)
    dots = (normals @ sv.astype(np.float32))  # (N,)
    flip = dots < 0.0
    normals_flipped = normals.copy()
    normals_flipped[flip] *= -1.0
    return normalize_rows(normals_flipped.astype(np.float64)).astype(np.float32)


# ------------------------------
# Mean Shift on Unit Sphere (MMS-like)
# ------------------------------

def mean_shift_on_sphere(normals: np.ndarray, R_deg: float, Ts_deg: float,
                         max_iter: int = 50, batch: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """Perform mean-shift on the unit sphere using chordal neighborhoods.
    Parameters
    ----------
    normals : (M,3) unit vectors (float32/64)
    R_deg   : angular search radius in degrees
    Ts_deg  : stop threshold in degrees (angle change per iteration)

    Returns
    -------
    modes   : (M,3) unit vectors (final MS destinations)
    labels  : (M,) int cluster labels after center merging (done later by merge_centers_by_angle)
              Here we just return placeholder labels = np.arange(M) (actual clustering is separate).
    """
    M = normals.shape[0]
    R_rad = deg2rad(R_deg)
    Ts_rad = deg2rad(Ts_deg)
    r_chord = angular_radius_to_chordal_radius(R_rad)

    tree = cKDTree(normals)
    modes = normals.copy().astype(np.float64)

    logging.info(
        f"Mean-shift on sphere: M={M}, R={R_deg}deg (chord {r_chord:.4f}), Ts={Ts_deg}deg, max_iter={max_iter}")

    for s in range(0, M, batch):
        e = min(s + batch, M)
        # vectorized neighbor queries for a batch of seeds
        seeds = modes[s:e]
        neigh_lists = tree.query_ball_point(seeds, r=r_chord)  # list-of-arrays for the whole batch
        for j, idxs in enumerate(neigh_lists):
            v = seeds[j]
            if len(idxs) == 0:
                modes[s + j] = v
                continue
            for _ in range(max_iter):
                m = normals[idxs].sum(axis=0)
                norm_m = np.linalg.norm(m)
                if norm_m < 1e-12:
                    break
                m = m / norm_m
                if angle_between(v, m) < Ts_rad:
                    v = m
                    break
                v = m
                # refresh neighbors occasionally to avoid Python overhead every iter
            modes[s + j] = v
        logging.debug(f"Mean-shift batch {s}:{e} / {M}")

    # Placeholder labels (clustering by angle performed next)
    labels = np.arange(M, dtype=np.int32)
    return modes.astype(np.float32), labels


def merge_centers_by_angle(modes: np.ndarray, Tm_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy merge of mode vectors by angular threshold Tm (degrees).
    Returns merged centers and labels for each mode.
    """
    Tm_rad = deg2rad(Tm_deg)
    centers: List[np.ndarray] = []
    labels = np.full(modes.shape[0], -1, dtype=np.int32)

    for i, v in enumerate(modes):
        assigned = False
        for cid, c in enumerate(centers):
            if angle_between(v, c) <= Tm_rad:
                labels[i] = cid
                # Update center as normalized mean
                c_new = normalize_rows((c[None, :] + v[None, :]))[0]
                centers[cid] = c_new
                assigned = True
                break
        if not assigned:
            centers.append(v.astype(np.float64))
            labels[i] = len(centers) - 1

    centers_np = normalize_rows(np.vstack(centers)).astype(np.float32)
    return centers_np, labels


def assign_labels_to_centers(normals: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each normal to the nearest center by maximum cosine (fast, vectorized)."""
    # normals: (M,3), centers: (C,3)
    sims = normals @ centers.T  # (M, C)
    labels = sims.argmax(axis=1).astype(np.int32)
    return labels


def mean_shift_landmark(normals: np.ndarray, R_deg: float, Ts_deg: float, Tm_deg: float,
                        sample_frac: float = 0.15, seed_max: int = 50000,
                        max_iter: int = 30, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Faster variant: run mean-shift on a subset of seeds, merge to centers, then assign all points.
    Suitable for very large M (e.g., >200k)."""
    rng = np.random.default_rng(random_state)
    M = normals.shape[0]
    S = min(seed_max, max(1000, int(sample_frac * M)))
    seed_idx = rng.choice(M, size=S, replace=False)
    logging.info(f"Landmark mean-shift: seeds={S}/{M} (sample_frac={sample_frac}, seed_max={seed_max})")

    seed_modes, _ = mean_shift_on_sphere(normals[seed_idx], R_deg=R_deg, Ts_deg=Ts_deg, max_iter=max_iter)
    centers, _ = merge_centers_by_angle(seed_modes, Tm_deg=Tm_deg)
    labels_all = assign_labels_to_centers(normals, centers)
    return centers, labels_all


def mean_shift_grid(normals: np.ndarray, bin_deg: float = 6.0, Tm_deg: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Very fast seeding: quantize directions by (azimuth,elevation) grid, use per-bin means as centers,
    then merge and assign. Good first pass when MMS full is too slow."""
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    az = np.arctan2(y, x)  # [-pi, pi]
    el = np.arcsin(np.clip(z, -1.0, 1.0))  # [-pi/2, pi/2]
    q = deg2rad(bin_deg)
    az_bin = np.floor((az + math.pi) / q).astype(np.int32)
    el_bin = np.floor((el + math.pi / 2.0) / q).astype(np.int32)
    keys = az_bin.astype(np.int64) << 32 | (el_bin.astype(np.int64) & 0xffffffff)
    # group by key
    order = np.argsort(keys)
    keys_sorted = keys[order]
    splits = np.where(np.diff(keys_sorted) != 0)[0] + 1
    groups = np.split(order, splits)
    centers = []
    for g in groups:
        c = normals[g].mean(axis=0)
        n = np.linalg.norm(c)
        if n < 1e-12:
            continue
        centers.append((c / n).astype(np.float32))
    if len(centers) == 0:
        centers = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
    centers = np.vstack(centers)
    centers, _ = merge_centers_by_angle(centers, Tm_deg=Tm_deg)
    labels = assign_labels_to_centers(normals, centers)
    return centers, labels


# ------------------------------
# Group post-processing: remove small clusters & flatness filter
# ------------------------------

def labels_to_groups(labels: np.ndarray) -> Dict[int, np.ndarray]:
    groups: Dict[int, np.ndarray] = {}
    for gid in np.unique(labels):
        if gid < 0:
            continue
        groups[gid] = np.where(labels == gid)[0]
    return groups


def remove_small_groups(groups: Dict[int, np.ndarray], min_size: int) -> Dict[int, np.ndarray]:
    return {gid: idxs for gid, idxs in groups.items() if idxs.size >= min_size}


def flatness_filter(normals: np.ndarray, groups: Dict[int, np.ndarray], centers: np.ndarray,
                    Tf_deg: float) -> Dict[int, np.ndarray]:
    """Within each group, keep points whose normal angle to group center <= Tf_deg."""
    Tf_rad = deg2rad(Tf_deg)
    cos_th = math.cos(Tf_rad)
    new_groups: Dict[int, np.ndarray] = {}
    for gid, idxs in groups.items():
        c = centers[gid]
        dots = normals[idxs] @ c
        keep = dots >= cos_th
        kept = idxs[keep]
        if kept.size > 0:
            new_groups[gid] = kept
    return new_groups


# ------------------------------
# Plane fitting
# ------------------------------

def fit_plane_svd(points: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Fit plane Ax+By+Cz+D=0 via SVD; return normalized (A,B,C) with ||n||=1 and D accordingly.
    Returns None if <3 points or degenerate.
    """
    if points.shape[0] < 3:
        return None
    centroid = points.mean(axis=0)
    X = points - centroid
    cov = X.T @ X / float(points.shape[0])
    w, V = np.linalg.eigh(cov)
    n = V[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    D = -float(n @ centroid)
    A, B, C = float(n[0]), float(n[1]), float(n[2])
    return A, B, C, D


# ------------------------------
# Heuristic for DBSCAN eps per-group
# ------------------------------

def heuristic_eps(points: np.ndarray, k: int = 5, quantile: float = 0.5) -> float:
    if points.shape[0] <= k:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # include self; take k-th neighbor
    kth = dists[:, k]
    return float(np.quantile(kth, quantile))


# ------------------------------
# Main pipeline
# ------------------------------

@dataclass
class Params:
    k: int = 30  # KNN for normals/curvature
    gmax: float = 0.06  # curvature threshold
    R_deg: float = 15.0  # mean-shift search radius (deg)
    Ts_deg: float = 0.5  # mean-shift stop threshold (deg)
    Tm_deg: float = 5.0  # mode merge angle (deg)
    C1_ratio: float = 0.01  # min group size ratio
    Tf_deg: float = 10.0  # flatness threshold (deg)
    eps: float = -1.0  # DBSCAN eps; if <0, auto per-group
    C2: int = 50  # DBSCAN min_samples
    batch_normals: int = 50000  # batch size for normals/curvature
    max_iter_ms: int = 50  # max iterations for mean shift
    # fast mean-shift options
    ms_mode: str = "auto"  # one of {auto, full, landmark, grid}
    ms_sample: float = 0.15  # fraction of seeds for landmark mode
    ms_seed_max: int = 50000  # max seeds for landmark mode
    ms_bin_deg: float = 6.0  # grid bin size (deg) for grid mode


def run_pipeline(input_ply: str, output_txt: str, params: Params, sensor_vec: Optional[np.ndarray] = None,
                 verbose: bool = False):
    setup_logger(verbose)
    logging.info("Loading PLY: %s", input_ply)
    P = read_ply_vertices(input_ply)  # (N,3)
    N = P.shape[0]
    logging.info(f"Loaded {N} points")

    # KNN indices for normals/curvature
    logging.info("KNN search (k=%d) ...", params.k)
    knn_idx = compute_knn_indices(P, k=params.k)

    # Normals + Curvature
    normals, curvature = compute_normals_and_curvature(P, knn_idx, batch=params.batch_normals, verbose=verbose)

    # Regulate normals
    normals = regulate_normals(normals, sensor_vec)

    # Curvature filter
    keep_mask = curvature <= params.gmax
    kept_idx = np.where(keep_mask)[0]
    dropped_idx = np.where(~keep_mask)[0]
    logging.info(f"Curvature filter g <= {params.gmax}: kept {kept_idx.size}, dropped {dropped_idx.size}")

    if kept_idx.size == 0:
        logging.warning("No points remain after curvature filtering; writing output with -1 labels.")
        write_output(P, normals, curvature, np.full(N, -1, int), np.full(N, -1, int), np.full(N, -1, int),
                     np.full((N, 4), np.nan, float), output_txt)
        return

    # Mean-shift on sphere (kept points only)
    N_kept = kept_idx.size

    # Choose MS mode
    ms_mode = params.ms_mode
    if ms_mode == 'auto':
        ms_mode = 'landmark' if N_kept > 200000 else 'full'
    logging.info(f"MS mode: {ms_mode}")

    normals_kept = normals[kept_idx]
    if ms_mode == 'full':
        modes, _ = mean_shift_on_sphere(normals_kept, R_deg=params.R_deg, Ts_deg=params.Ts_deg,
                                        max_iter=params.max_iter_ms)
        centers, labels_kept = merge_centers_by_angle(modes, Tm_deg=params.Tm_deg)
    elif ms_mode == 'landmark':
        centers, labels_kept = mean_shift_landmark(normals_kept, R_deg=params.R_deg, Ts_deg=params.Ts_deg,
                                                   Tm_deg=params.Tm_deg, sample_frac=params.ms_sample,
                                                   seed_max=params.ms_seed_max, max_iter=params.max_iter_ms)
    elif ms_mode == 'grid':
        centers, labels_kept = mean_shift_grid(normals_kept, bin_deg=params.ms_bin_deg, Tm_deg=params.Tm_deg)
    else:
        raise ValueError(f"Unknown ms_mode: {ms_mode}")

    # Merge centers by angle -> centers + labels for kept points
    # (already done per mode)
    # centers, labels_kept = merge_centers_by_angle(modes, Tm_deg=params.Tm_deg)

    # Build groups from kept labels
    groups_kept = labels_to_groups(labels_kept)

    # Remove small groups (size < C1_ratio * kept)
    min_size = max(1, int(params.C1_ratio * N_kept))
    min_size = 100
    groups_kept = remove_small_groups(groups_kept, min_size=min_size)
    logging.info(f"Groups after removal (min_size={min_size}): {len(groups_kept)}")

    # Flatness filter within each group
    groups_kept = flatness_filter(normals[kept_idx], groups_kept, centers, Tf_deg=params.Tf_deg)

    # Prepare per-point outputs (default unassigned)
    cluster_id = np.full(N, -1, dtype=np.int32)
    subcluster_id = np.full(N, -1, dtype=np.int32)
    facet_id = np.full(N, -1, dtype=np.int32)
    plane_params = np.full((N, 4), np.nan, dtype=np.float64)

    # Assign cluster_id to kept points that passed flatness
    # Map from kept local index -> global index
    kept_global = kept_idx

    # Reindex groups to contiguous group IDs
    remap_gid = {g_old: g_new for g_new, g_old in enumerate(sorted(groups_kept.keys()))}

    # Iterate groups -> DBSCAN per group
    facet_counter = 0
    for g_old, local_indices in groups_kept.items():
        gid = remap_gid[g_old]
        global_indices = kept_global[local_indices]
        cluster_id[global_indices] = gid

        # DBSCAN on spatial coordinates of points in this group
        pts_g = P[global_indices]
        if pts_g.shape[0] < params.C2:
            logging.debug(f"Group {gid}: not enough points for DBSCAN (|G|={pts_g.shape[0]} < C2={params.C2})")
            continue
        if params.eps > 0:
            eps_g = params.eps
        else:
            eps_g = 1.5 * heuristic_eps(pts_g, k=5, quantile=0.5)
            eps_g = max(eps_g, 1e-6)
        db = DBSCAN(eps=eps_g, min_samples=params.C2)
        labels_db = db.fit_predict(pts_g)

        n_sub = int(labels_db.max() + 1) if labels_db.size > 0 else 0
        logging.info(f"Group {gid}: DBSCAN eps={eps_g:.4f} -> {n_sub} subclusters")

        for sid in range(n_sub):
            mask_sid = labels_db == sid
            idx_sid_global = global_indices[mask_sid]
            subcluster_id[idx_sid_global] = sid
            # Fit plane for this subcluster
            plane = fit_plane_svd(P[idx_sid_global])
            if plane is None:
                continue
            A, B, C, D = plane
            plane_params[idx_sid_global, 0] = A
            plane_params[idx_sid_global, 1] = B
            plane_params[idx_sid_global, 2] = C
            plane_params[idx_sid_global, 3] = D
            # Assign facet_id
            facet_id[idx_sid_global] = facet_counter
            facet_counter += 1

    # Write output
    write_output(P, normals, curvature, cluster_id, subcluster_id, facet_id, plane_params, output_txt)


def write_output(P: np.ndarray, normals: np.ndarray, curvature: np.ndarray,
                 cluster_id: np.ndarray, subcluster_id: np.ndarray, facet_id: np.ndarray,
                 plane_params: np.ndarray, output_txt: str):
    header = ["X", "Y", "Z", "cluster_id", "subcluster_id", "facet_id", "A", "B", "C", "D", "curvature"]
    # Assemble data matrix in required order; ints must be converted to float for consistent formatting
    # We'll cast the int columns when formatting via write_txt_tsv's fmt string
    data = np.column_stack([
        P[:, 0], P[:, 1], P[:, 2],
        cluster_id.astype(np.int64),
        subcluster_id.astype(np.int64),
        facet_id.astype(np.int64),
        plane_params[:, 0], plane_params[:, 1], plane_params[:, 2], plane_params[:, 3],
        curvature.astype(np.float64)
    ])
    write_txt_tsv(output_txt, header, data)
    logging.info("Wrote TXT: %s", output_txt)


# ------------------------------
# CLI
# ------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="MMS-DBSCAN discontinuity extraction with plane params and curvature")
    p.add_argument('-i', '--input', required=True, help='Input point cloud (.ply)')
    p.add_argument('-o', '--output', required=True, help='Output TXT (TSV) path')
    p.add_argument('--k', type=int, default=30, help='KNN for normals/curvature (default: 30)')
    p.add_argument('--gmax', type=float, default=0.06, help='Curvature threshold g<=gmax to keep (default: 0.06)')
    p.add_argument('--R', type=float, default=15.0, help='Mean-shift angular radius in degrees (default: 15)')
    p.add_argument('--Ts', type=float, default=0.5, help='Mean-shift stop threshold (deg) (default: 0.5)')
    p.add_argument('--Tm', type=float, default=5.0, help='Mode-merge angle (deg) (default: 5)')
    p.add_argument('--C1', type=float, default=0.01, help='Min group size ratio (default: 0.01)')
    p.add_argument('--Tf', type=float, default=10.0, help='Flatness angle threshold (deg) (default: 10)')
    p.add_argument('--eps', type=float, default=-1.0, help='DBSCAN eps in XYZ units; <0 to auto (default: -1)')
    p.add_argument('--C2', type=int, default=50, help='DBSCAN min_samples (default: 50)')
    p.add_argument('--batch', type=int, default=50000, help='Batch size for normals/curvature (default: 50000)')
    p.add_argument('--ms-iter', type=int, default=50, help='Max iterations for mean shift (default: 50)')
    p.add_argument('--sensor', type=float, nargs=3, default=None, metavar=('SX', 'SY', 'SZ'),
                   help='Optional sensor vector for normal regulation (default: +Z)')
    # fast MMS options
    p.add_argument('--ms-mode', type=str, default='auto', choices=['auto', 'full', 'landmark', 'grid'],
                   help='Mean-shift mode: auto/full/landmark/grid (default: auto)')
    p.add_argument('--ms-sample', type=float, default=0.15,
                   help='Landmark mode: sampling fraction (default: 0.15)')
    p.add_argument('--ms-seed-max', type=int, default=50000,
                   help='Landmark mode: max number of seeds (default: 50000)')
    p.add_argument('--ms-bin-deg', type=float, default=6.0,
                   help='Grid mode: bin size in degrees (default: 6.0)')
    return p


def main():
    args = build_argparser().parse_args()
    params = Params(
        k=args.k,
        gmax=args.gmax,
        R_deg=args.R,
        Ts_deg=args.Ts,
        Tm_deg=args.Tm,
        C1_ratio=args.C1,
        Tf_deg=args.Tf,
        eps=args.eps,
        C2=args.C2,
        batch_normals=args.batch,
        max_iter_ms=args.ms_iter,
    )
    sensor_vec = np.array(args.sensor, dtype=np.float64) if args.sensor is not None else None
    run_pipeline(args.input, args.output, params, sensor_vec=sensor_vec, verbose=True)


if __name__ == '__main__':
    main()

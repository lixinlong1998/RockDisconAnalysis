# -*- coding: utf-8 -*-
"""
ExportBlockShellPolygons.py
把每个“外壳（候选块体）”对应的结构面 polygon 合并到一个 PLY（仅线框，多边形边）。
依赖：src/Export.py 中的 export_to_meshlab_ply、polygon_vertexs_to_edges；
      Discontinuity 对象需具备 polygon_vertex_fit_plane、get_polygon()、type 等字段/方法。
"""

import os
import numpy as np
from typing import List
from src import Export

# 颜色方案（RGB 0-255）
COLOR_FREE  = np.array([0, 200, 0],   dtype=np.uint8)   # freeface：绿色
COLOR_JOINT = np.array([30, 144, 255],dtype=np.uint8)   # jointface：天蓝
COLOR_UNDEF = np.array([180, 180, 180],dtype=np.uint8)  # undefined：灰

def _ensure_polygon_ready(disc, method: str = 'convex'):
    """确保多边形已计算；若没有则按指定 method 计算。"""
    coords = getattr(disc, 'polygon_vertex_fit_plane', None)
    if coords is None or (isinstance(coords, np.ndarray) and coords.shape[0] < 3):
        try:
            disc.get_polygon(method=method)
        except Exception:
            return None
        coords = getattr(disc, 'polygon_vertex_fit_plane', None)
    return coords

def _color_for_type(disc_type: str):
    if disc_type == 'freeface':
        return COLOR_FREE
    elif disc_type == 'jointface':
        return COLOR_JOINT
    else:
        return COLOR_UNDEF

def export_block_shell_polygons(save_dir: str,
                                blocks: List,
                                discontinuitys,
                                polygon_method: str = 'convex'):
    """
    将每个 block（外壳）对应的结构面的 polygon 组合输出为单独的 PLY（线框）。
    参数
    ----
    save_dir : 输出目录
    blocks : List[BlockResult]（要求每个 BlockResult 有 block_id 和 face_ids）
    discontinuitys : 你的 Discontinuitys 集合对象，含 .discontinuitys 列表
    polygon_method : 'convex' | 'ashape' ；当 polygon 尚未计算时使用的方法

    返回：每个生成的 ply 文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    disc_list = discontinuitys.discontinuitys
    out_paths = []

    for blk in blocks:
        vertices_all, edges_all, colors_all = [], [], []
        vtx_offset = 0

        for idx in blk.face_ids:
            disc = disc_list[idx]
            coords = _ensure_polygon_ready(disc, method=polygon_method)
            if coords is None or len(coords) < 2:
                continue

            coords = np.asarray(coords, dtype=np.float32)
            n = coords.shape[0]
            vertices_all.append(coords)

            # 边（索引需加偏移）
            poly_edges = Export.polygon_vertexs_to_edges(coords)
            if poly_edges is not None and len(poly_edges) > 0:
                poly_edges = np.asarray(poly_edges, dtype=np.int32) + vtx_offset
                edges_all.append(poly_edges)

            # 颜色
            color = _color_for_type(getattr(disc, 'type', 'undefined'))
            colors_all.append(np.tile(color, (n, 1)))

            vtx_offset += n

        if len(vertices_all) == 0:
            continue

        vertices_all = np.vstack(vertices_all).astype(np.float32)
        colors_all   = np.vstack(colors_all).astype(np.uint8)
        edges_all    = np.vstack(edges_all).astype(np.int32) if len(edges_all) > 0 else None

        save_path = os.path.join(save_dir, f"BlockShell_{blk.block_id}_polygons.ply")
        Export.export_to_meshlab_ply(save_path, vertices=vertices_all, edges=edges_all, faces=None, colors=colors_all)
        out_paths.append(save_path)

    return out_paths

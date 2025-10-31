 import struct
 import os
 import numpy as np


 def export_each_combination(save_dir_path, discontinuitys):
    """
    将每个结构面对应的内容组合导出到 PLY 文件（MeshLab 可视化）。
    包含：
        - 内点（inliers，白色）
        - 离群点（outliers，灰色）
        - 多边形边界线（随机颜色）
        - 迹线（黄色）
        - 椭圆盘模型（随机颜色）
        - 法向线（红色）

    Parameters
    ----------
    save_dir_path : str
        输出文件夹路径，path_visualize_combinations
    discontinuitys : object
        Discontinuitys 集合，包含多个 Discontinuity 对象
    save_path : str
        输出文件夹路径，每个结构面会生成一个 PLY 文件
    """
    resolution = 60
    PI2 = 2 * np.pi
    np.random.seed(0)

    for disc in discontinuitys.discontinuitys:

        if disc.valid is False:
            # print(f"[警告] 结构面无效，跳过 {disc.joint_id}-{disc.joint_cluster_id}.")
            continue

        vertices_all = []
        edges_all = []
        faces_all = []
        colors_all = []
        vertex_offset = 0

        # 1. 内点 (绿色)
        rock_points = np.array([p.coord for p in disc.rock_points.points], dtype=np.float32)
        inlier_coords = rock_points[disc.inlier_mask]
        inlier_colors = np.tile([0, 255, 0], (len(inlier_coords), 1))
        vertices_all.extend(inlier_coords)
        colors_all.extend(inlier_colors)
        vertex_offset += len(inlier_coords)

        # 2. 离群点 (灰色)
        if np.any(~disc.inlier_mask):
            outlier_coords = rock_points[~disc.inlier_mask]
            outlier_colors = np.tile([128, 128, 128], (len(outlier_coords), 1))
            vertices_all.extend(outlier_coords)
            vertex_offset += len(outlier_coords)
            colors_all.extend(outlier_colors)

        # 3. 多边形边界 (黑色)
        polygon_coords = disc.polygon_vertex_fit_plane
        random_color = np.random.randint(0, 256, 3)
        poly_colors = np.tile(random_color, (len(polygon_coords), 1))
        poly_edges = polygon_vertexs_to_edges(polygon_coords) + vertex_offset
        vertices_all.extend(polygon_coords)
        vertex_offset += len(polygon_coords)
        colors_all.extend(poly_colors)
        edges_all.extend(poly_edges)

        # 4. 迹线 (黄色)
        trace_coords = np.array([disc.trace_vertex_1, disc.trace_vertex_2], dtype=np.float32)
        trace_colors = np.tile([255, 255, 0], (2, 1))  # 黄
        trace_edges = np.array([[0, 1]]) + vertex_offset
        vertices_all.extend(trace_coords)
        vertex_offset += len(trace_coords)
        colors_all.extend(trace_colors)
        edges_all.extend(trace_edges)

        # 5. 椭圆盘 (随机色)
        center = disc.disc_center
        # 长短轴向量及长度
        a_norm = disc.long_axis_norm
        b_norm = disc.short_axis_norm
        a_len = disc.ellip_a
        b_len = disc.ellip_b
        # 圆周角度
        angles = np.linspace(0, PI2, resolution, endpoint=False)
        boundary_pts = [center + a_len * np.cos(t) * a_norm + b_len * np.sin(t) * b_norm for t in angles]
        # 顶点顺序：先中心点，再边界点
        disk_coords = np.vstack([center, boundary_pts])
        vertices_all.extend(disk_coords)
        # 顶点赋色
        disk_color = np.random.randint(0, 256, 3)
        disk_colors = np.tile(disk_color, (len(disk_coords), 1))
        colors_all.extend(disk_colors)
        # 生成三角面：center与每一对相邻边界点构成
        for i in range(resolution):
            i1 = vertex_offset + 0  # center
            i2 = vertex_offset + 1 + i
            i3 = vertex_offset + 1 + ((i + 1) % resolution)
            faces_all.append([i1, i2, i3])
        vertex_offset += len(disk_coords)

        # 6. 法向线 (红色)
        norm_start = disc.disc_center
        norm_end = norm_start + disc.disc_normal * disc.ellip_b
        norm_coords = np.array([norm_start, norm_end], dtype=np.float32)
        norm_colors = np.tile([255, 0, 0], (2, 1))  # 红
        norm_edges = np.array([[0, 1]]) + vertex_offset
        vertices_all.extend(norm_coords)
        vertex_offset += len(norm_coords)
        colors_all.extend(norm_colors)
        edges_all.extend(norm_edges)

        # 保存当前组合
        vertices_all = np.array(vertices_all, dtype=np.float32)
        colors_all = np.array(colors_all, dtype=np.uint8)
        edges_all = np.array(edges_all, dtype=np.int32) if edges_all else None
        faces_all = np.array(faces_all, dtype=np.int32) if faces_all else None
        filename = os.path.join(save_dir_path, f"combination_{disc.joint_id}_{disc.joint_cluster_id}.ply")
        export_to_meshlab_ply(filename, vertices=vertices_all, edges=edges_all, faces=faces_all, colors=colors_all)
        # print(f"[完成] 已导出结构面 {disc.joint_id}-{disc.joint_cluster_id} 到 {filename}")


def export_each_clusterpoints(clusters, visualize_path):
    for cluster in clusters:
        points = cluster.rock_points.points
        coords = [point.coord for point in points]
        save_path = os.path.join(visualize_path,
                                 f'cluster_points_{cluster.joint_id}_{cluster.joint_cluster_id}.ply')
        export_to_meshlab_ply(save_path, vertices=coords)


def export_discontinuity_trace(discontinuitys, save_path):
    """
    将 trace_dict 中的每条迹线（start, end）导出为包含 line 类型的 PLY 文件

    Parameters
    ----------
    trace_dict : dict
        每条迹线包含 'start', 'end' 坐标
    filename : str
        输出 ply 文件路径
    """

    points = []
    edges = []
    failed_cont = 0
    for i, disc in enumerate(discontinuitys.discontinuitys):
        if disc.trace_vertex_1 is None or disc.trace_vertex_2 is None:
            failed_cont += 1
            continue

        # get coord
        start = disc.trace_vertex_1
        end = disc.trace_vertex_2

        # write data
        idx_start = len(points)
        idx_end = idx_start + 1
        points.append(start)
        points.append(end)
        edges.append([idx_start, idx_end])

    points = np.array(points, dtype=np.float64)
    edges = np.array(edges, dtype=np.int32)

    write_ply_with_edges(save_path, points, edges)
    print(f"{edges.shape[0]}个迹线段成功(失败{failed_cont}个)导出到 {save_path}")


def export_segments(segments: list, save_path):
    """
    将 segments 中的每条Segment.Segment对象的start, end导出为包含 line 类型的 PLY 文件

    Parameters
    ----------
    segments : list
        每条交线段包含 'start', 'end' 坐标
    filename : str
        输出 ply 文件路径
    """

    points = []
    edges = []
    failed_cont = 0
    for i, segment in enumerate(segments):
        start = segment.p1
        end = segment.p2

        # write data
        idx_start = len(points)
        idx_end = idx_start + 1
        points.append(start)
        points.append(end)
        edges.append([idx_start, idx_end])

    points = np.array(points, dtype=np.float64)
    edges = np.array(edges, dtype=np.int32)

    write_ply_with_edges(save_path, points, edges)
    print(f"{edges.shape[0]}个迹线段成功(失败{failed_cont}个)导出到 {save_path}")


def write_ply_with_edges(filename, points, edges):
    """
    自定义写入 PLY 文件，包含 vertex 和 edge（line）

    Parameters
    ----------
    filename : str
        输出 ply 文件名
    points : (N, 3) np.ndarray
        顶点数组，float32
    edges : (M, 2) np.ndarray
        每条线由两个点索引组成，int32
    """
    num_vertices = points.shape[0]
    num_edges = edges.shape[0]

    with open(filename, 'wb') as f:
        # ===== 写入 PLY 头部 =====
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
element edge {num_edges}
property int vertex1
property int vertex2
end_header
"""
        f.write(header.encode('utf-8'))

        # ===== 写入 vertex 坐标数据 (float32) =====
        for pt in points:
            f.write(struct.pack('<fff', *pt))

        # ===== 写入 edge 数据（每条线是两个顶点索引） =====
        for edge in edges:
            f.write(struct.pack('<ii', *edge))


def export_discontinuity_ellipdisk(discontinuitys, save_path, resolution=100):
    '''
    将多个椭圆面导出为 ply 文件，可在 MeshLab 中可视化。
    参数:
        file_path (str): 保存的 ply 文件路径，例如 'discs.ply'
        discs (List[Dict]): 每个椭圆用一个字典表示，包含以下字段:
            - 'center' (np.ndarray, shape=(3,))
            - 'long_axis_vertex' (np.ndarray, shape=(2, 3)) → [a1, a2]
            - 'short_axis_vertex' (np.ndarray, shape=(2, 3)) → [b1, b2]
            - 'normal' (np.ndarray, shape=(3,))
        resolution (int): 圆盘边界点的数量（越高越圆滑）
    返回:
        无，直接写入 ply 文件
    '''
    vertices = []
    faces = []
    vert_id_offset = 0  # 全局顶点索引偏移
    failed_count = 0
    success_count = 0

    PI2 = 2 * np.pi
    for disc in discontinuitys.discontinuitys:
        if not disc.valid:
            failed_count += 1
            continue
        else:
            success_count += 1

        center = disc.disc_center

        # 长短轴向量及长度
        a_norm = disc.long_axis_norm
        b_norm = disc.short_axis_norm
        a_len = disc.ellip_a
        b_len = disc.ellip_b

        # 圆周角度
        angles = np.linspace(0, PI2, resolution, endpoint=False)
        boundary_pts = [center + a_len * np.cos(theta) * a_norm + b_len * np.sin(theta) * b_norm for theta in angles]

        # 顶点顺序：先中心点，再边界点
        local_verts = [center] + boundary_pts
        vertices.extend(local_verts)

        # 生成三角面：center与每一对相邻边界点构成
        for i in range(resolution):
            i1 = vert_id_offset + 0  # center
            i2 = vert_id_offset + 1 + i
            i3 = vert_id_offset + 1 + ((i + 1) % resolution)
            faces.append([i1, i2, i3])

        vert_id_offset += len(local_verts)

    # 写入 ply 文件（ascii 格式）
    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    print(f"{success_count}个椭圆面成功(失败{failed_count}个)导出到 {save_path}")


def polygon_to_edges(faces):
    """
    将多边形面片拆解成边（自动去重）
    :param faces: 面片数组，shape=(K, N)，每行是 [v1, v2, ..., vN]（N >= 3）
    :return: 边数组，shape=(M, 2)，每行是 [v_start, v_end]
    """
    edges = set()  # 用集合去重
    for face in faces:
        n = len(face)
        for i in range(n):
            v1 = face[i]
            v2 = face[(i + 1) % n]  # 循环连接最后一个顶点到第一个顶点
            edge = tuple(sorted((v1, v2)))  # 排序避免重复（如 (0,1) 和 (1,0)）
            edges.add(edge)
    return np.array(list(edges), dtype=np.int32)


def polygon_vertexs_to_edges(vertex_coords):
    """
    将给定三维坐标序列拆解成边（自动去重）
    :param vertex_coords: 顶点序列数组，shape=(K, 3)
    :return: 边数组，shape=(M, 2)，每行是 [v_start, v_end]
    """
    n = len(vertex_coords)
    edges = set()  # 用集合去重
    for i in range(n):
        v1 = i
        v2 = (i + 1) % n
        edge = tuple(sorted((v1, v2)))  # 排序避免重复（如 (0,1) 和 (1,0)）
        edges.add(edge)
    return np.array(list(edges), dtype=np.int32)


def export_discontinuity_polygon(discontinuitys, save_path):
    # visualize for each discontinuity
    visualize_path = os.path.join(os.path.dirname(save_path), "visualize", "DisconPolygons")
    os.makedirs(visualize_path, exist_ok=True)

    # 用于合并所有多边形的数据结构
    all_vertices = []  # 所有顶点坐标
    all_edges = []  # 所有边（已偏移索引）
    all_colors = []  # 所有顶点颜色
    added_vertex_count = 0

    # 1. 首先处理每个不连续面并生成单独的PLY文件
    for discon in discontinuitys.discontinuitys:
        # 当前多边形的顶点坐标
        vertex_coords = discon.polygon_vertex_fit_plane
        vertex_n = vertex_coords.shape[0]

        # 为当前多边形生成随机颜色
        random_color = np.random.randint(0, 256, 3)
        vertex_colors = np.tile(random_color, (vertex_n, 1))

        # 将多边形面片拆解成边（自动去重）,索引值从0开始到len(vertex_coords)
        polygon_edges = polygon_vertexs_to_edges(vertex_coords)

        # 为每个多边形单独保存文件
        save_path_multi = os.path.join(visualize_path, f'polygon_{discon.joint_id}_{discon.joint_cluster_id}.ply')
        export_to_meshlab_ply(save_path_multi,
                              vertices=vertex_coords,
                              edges=polygon_edges,
                              faces=None,
                              colors=vertex_colors)

        # 将当前多边形添加到合并列表
        all_vertices.extend(vertex_coords)
        all_colors.extend(vertex_colors)

        # 计算边的偏移索引
        for edge in polygon_edges:
            current_vertex_id1, current_vertex_id2 = edge
            v1 = added_vertex_count + current_vertex_id1
            v2 = added_vertex_count + current_vertex_id2
            all_edges.append([v1, v2])

        added_vertex_count += vertex_n

    # 2. 将所有多边形合并到一个PLY文件
    if all_vertices:
        # 转换类型为NumPy数组
        all_vertices = np.array(all_vertices, dtype=np.float32)
        all_edges = np.array(all_edges, dtype=np.int32)
        all_colors = np.array(all_colors, dtype=np.int32)

        # 导出合并后的文件
        export_to_meshlab_ply(save_path,
                              vertices=all_vertices,
                              edges=all_edges,
                              faces=None,
                              colors=all_colors)

        print(f"已导出 {len(discontinuitys.discontinuitys)} 个多边形到 {save_path}")
    else:
        print("未找到多边形数据，未生成文件")


def export_to_meshlab_ply(filename, vertices=None, edges=None, faces=None, colors=None):
    """
    导出点、线、面到 PLY 文件，支持 MeshLab 可视化
    :param filename: 输出文件名（如 "output.ply"）
    :param vertices: 顶点数组，shape=(N, 3)，每行是 [x, y, z]
    :param edges: 边数组，shape=(M, 2)，每行是 [vertex_idx1, vertex_idx2]
    :param faces: 面数组，shape=(K, 3)，每行是 [vertex_idx1, vertex_idx2, vertex_idx3]
    :param colors: 顶点颜色，shape=(N, 3)，每行是 [r, g, b]（0-255）
    """
    if vertices is None:
        raise ValueError("顶点数据不能为空！")

    vertices = np.asarray(vertices, dtype=np.float32)
    has_edges = edges is not None
    has_faces = faces is not None
    has_colors = colors is not None

    with open(filename, 'w') as f:
        # 写入 PLY 头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if has_edges:
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        if has_faces:
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # 写入顶点数据（+颜色）
        for i, v in enumerate(vertices):
            line = f"{v[0]} {v[1]} {v[2]}"
            if has_colors:
                c = colors[i]
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")

        # 写入边数据
        if has_edges:
            for e in edges:
                f.write(f"{e[0]} {e[1]}\n")

        # 写入面数据
        if has_faces:
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    # print(f"已导出到 {filename}，可用 MeshLab 打开！")

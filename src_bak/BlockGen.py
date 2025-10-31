import time
from scipy.spatial import cKDTree as KDTree  # 替换 sklearn 的 KDTree
import networkx as nx
from multiprocessing import cpu_count
from multiprocessing import Pool
from numba import njit, prange
from collections import defaultdict
import multiprocessing
import numpy as np
from src import Segment


def get_segments(discontinuitys, extention=1.5, pool_size=16, max_memory=32):
    '''
    :param discontinuitys:
    :param extention: 用于控制椭圆盘的缩放比例
    :return:segments =
    '''
    starttime = time.perf_counter()

    # load datalist
    datalist = discontinuitys.discontinuitys

    # 用disk的长轴建立球体，判断所有球体之间的相交情况，记录相交的两球体对应的disk pair
    substarttime = time.perf_counter()
    discontinuity_pairs = find_discontinuity_pairs(datalist, extention)
    print(f'[test]len(discontinuity_pairs):{len(discontinuity_pairs)}')
    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 基于球体相交得到discontinuity pair.')

    # 使用 NumPy 向量化实现两个结构面（Discontinuity）之间交线的方向向量快速求解;注意这里面可能存在共面或近乎共面的情况
    substarttime = time.perf_counter()
    line_dirs, points, coplanar_masks = intersect_plane(discontinuity_pairs)
    print(f'[test]line_dirs[0]:{line_dirs[0]}')
    print(f'[test]len(line_dirs):{len(line_dirs)}')
    print(f'[test]points[0]:{points[0]}')
    print(f'[test]len(points):{len(points)}')
    print(f'[test]coplanar_masks[0]:{coplanar_masks[0]}')
    print(f'[test]coplanar_masks中True的个数:{np.count_nonzero(coplanar_masks)}')
    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 求解两结构面之间交线的方向向量.')

    # 使用 NumPy 向量化将交线投影到结构面上求解交线段
    substarttime = time.perf_counter()
    intersection_pairs, valid_intersection_pairs = intersect_ellipdisk(discontinuity_pairs, line_dirs, points,
                                                                       coplanar_masks)

    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 求解交线与结构面模型的交线段.')

    # 采用并行运算来求解有限元交线段
    substarttime = time.perf_counter()
    # segments = multiprocess_extract_seg_from_intsec_pair(intersection_pairs, discontinuity_pairs, line_dirs, points,
    #                                                      pool_size=16)
    segments = extract_seg_from_intsec_pair(intersection_pairs, discontinuity_pairs, line_dirs, points,
                                            tol_relative=1e-6)
    print(f'[test]segments[0]:{segments[0]}')
    print(f'[test]len(segments):{len(segments)}')
    print(f'[test]segments[0].surface_ids.:{segments[0].surface_ids}')
    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 交线段求并集.')

    print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — 求解有限元交线段.')
    return segments


def get_vertex_graph(discontinuitys, segments, extention=1.5, pool_size=16, max_memory=32):
    # 两两求解segments之间的交点node，考虑了浮点运算的截尾误差
    # nodes 是[(i, j, pt),...]
    substarttime = time.perf_counter()
    nodes = parallel_segment_intersections(segments, tol=1e-6, pool_size=pool_size)
    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 两两求解segments之间的交点node.')

    # 聚合node为unique node，因为一个node实际上至少对应了3组segment pairs
    substarttime = time.perf_counter()
    unodes_coord, unodes_by_disc = aggregate_nodes_with_tolerance(nodes, segments, tol=1e-6)
    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 聚合node为unique node.')

    # 构建有向图网络
    substarttime = time.perf_counter()
    vertex_graph = extract_surface_loops(unodes_coord, unodes_by_disc, segments, pool_size=None)
    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 构建有向图网络.')
    return vertex_graph, unodes_coord


def get_rockblocks(vertex_graph, unodes_coord):
    """
    基于Half-Face类，提取闭合多面体块体。
    具体参数与功能同前，省略重复注释。
    """
    from collections import defaultdict
    import numpy as np

    def compute_face_normal(points):
        import numpy as np
        v1 = np.array(points[1]) - np.array(points[0])
        v2 = np.array(points[2]) - np.array(points[0])
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return np.array([0, 0, 0])
        return normal / norm

    def check_block_closed(block_faces):
        from collections import Counter
        edge_counter = Counter()
        for hf in block_faces:
            for edge in hf.edges():
                edge_counter[edge] += 1
        return all(count == 2 for count in edge_counter.values())

    # ############################################## main body #########################################################
    # 采用面-面邻接法 + 空间遍历法构建块体
    substarttime = time.perf_counter()

    edge_to_half_face = defaultdict(list)
    half_faces = []

    for cluster_id, loops in vertex_graph.items():
        for loop in loops:
            pt_indices = [np.where((unodes_coord == pt).all(axis=1))[0][0] for pt in loop]
            normal = compute_face_normal([unodes_coord[i] for i in pt_indices])
            hf = HalfFace(cluster_id, pt_indices, normal)
            hf_idx = len(half_faces)
            hf.index(hf_idx)  # 写入id
            half_faces.append(hf)

            for edge in hf.edges():
                edge_to_half_face[edge].append(hf_idx)

    visited = set()
    blocks = []
    half_face_to_block = {}

    for hf_idx, hf in enumerate(half_faces):
        if hf_idx in visited:
            continue

        block_faces = []
        stack = [hf_idx]

        while stack:
            current_idx = stack.pop()
            if current_idx in visited:
                continue

            visited.add(current_idx)
            current_hf = half_faces[current_idx]
            block_faces.append(current_hf)

            for edge in current_hf.edges():
                neighbors = edge_to_half_face[edge]
                for neighbor_idx in neighbors:
                    if neighbor_idx == current_idx:  # 还需要判断是否是另一个conuterpart
                        continue
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)

        if check_block_closed(block_faces):
            block_id = len(blocks)
            face_indices = [hf.pt_indices for hf in block_faces]
            blocks.append({
                'faces': face_indices,
                'adjacent_blocks': set(),
            })
            for bf in block_faces:
                hf_idx = half_face_to_index[bf]
                half_face_to_block[hf_idx] = block_id

    for edge, hfs in edge_to_half_face.items():
        if len(hfs) == 2:
            block_a = half_face_to_block.get(hfs[0])
            block_b = half_face_to_block.get(hfs[1])
            if block_a is not None and block_b is not None and block_a != block_b:
                blocks[block_a]['adjacent_blocks'].add(block_b)
                blocks[block_b]['adjacent_blocks'].add(block_a)

    for blk in blocks:
        blk['adjacent_blocks'] = list(blk['adjacent_blocks'])

    print(f'[time cost]{time_cost_hms(time.perf_counter() - substarttime)} — 采用面-面邻接法 + 空间遍历法构建块体.')

    return blocks


def find_discontinuity_pairs(datalist, extention):
    '''

    :param datalist:
    :param extention:
    :return:
    '''

    def ball_neighbour_search(balls):
        '''

        :param balls: np.array([[x, y, z, r], ...])
        :return:
        '''
        centers = balls[:, :3]
        radii = balls[:, 3]

        # 建立 KDTree 时间复杂度约 O(N log N)，适合上万规模
        tree = KDTree(centers)

        # 每个球搜索“可能相交”的近邻：搜索距离 = 最大可能半径之和
        max_radius = np.max(radii)
        pairs = set()
        for i in range(len(balls)):
            center_i = centers[i]
            radius_i = radii[i]
            # 搜索半径设为：r_i + max(r) ≈ 最坏情况
            # 逻辑：如果r_i + max(r)内都没有其他球心的话，那么该球没有neighbors
            neighbors = tree.query_ball_point(center_i, r=radius_i + max_radius)
            for j in neighbors:
                if i < j:
                    # 精确判断是否相交
                    dist_ij = np.linalg.norm(centers[i] - centers[j])
                    if dist_ij <= radii[i] + radii[j]:
                        pairs.add((i, j))

        return list(pairs)  # pairs = [(i, j), ...]  # 所有相交球体的索引对

    # ########################################### __main body of function__ ###########################################
    # 过滤invalid结构面
    valid_discs = []
    index_map = []  # 存储 valid_discs 中对应原始 datalist 的索引
    for i, d in enumerate(datalist):
        if d.disc_center is not None and d.ellip_a is not None:
            valid_discs.append(d)
            index_map.append(i)

    # 构建球体数组 (N, 4) # balls = np.array([[x, y, z, r], ...])  # shape: (N, 4)，中心坐标+半径
    balls = np.asarray([[*d.disc_center, d.ellip_a * extention] for d in valid_discs], dtype=np.float64)

    # 通过球体的球心和半径寻找相交球对
    pairs = ball_neighbour_search(balls)

    # 从pairs返回discontinuity类对象对
    discontinuity_pairs = [(datalist[index_map[i]], datalist[index_map[j]]) for i, j in pairs]

    return discontinuity_pairs


def intersect_ellipdisk(discontinuity_pairs, line_dirs, points, coplanar_masks):
    f"""
    计算一条空间直线与椭圆盘结构面的交点（最多2个）
    numpy向量版本，结合Numba，批量计算结构面交线
    :param discontinuity_pairs: List[Tuple[Discontinuity, Discontinuity]]
    :param line_dirs: ndarray, shape (m, 3)，每对结构面的交线方向
    :param points: ndarray, shape (m, 3)，每条交线上的一点
    :param coplanar_masks: ndarray, shape (m,)，共面掩码
    :return: intersection_pairs: Dict[pair_idx:[intersection, intersection], intersection = [t_min, t_max] or [None, None]
    """

    @njit(parallel=True)
    def batch_intersection_2d(disc_centers, long_axis_vectors, short_axis_vectors, a_vals, b_vals, points_on_line,
                              line_dirs):
        """
        Numba并行加速版本：批量计算交线投影到结构面二维坐标系后的交点参数区间 t1, t2

        :param disc_centers: ndarray, shape (N, 3)
        :param long_axis_vectors: ndarray, shape (N, 3)
        :param short_axis_vectors: ndarray, shape (N, 3)
        :param a_vals: ndarray, shape (N,)
        :param b_vals: ndarray, shape (N,)
        :param points_on_line: ndarray, shape (N, 3)
        :param line_dirs: ndarray, shape (N, 3)
        :return: t_min, t_max, delta_mask (N,)
        """

        N = disc_centers.shape[0]
        t_min = np.empty(N, dtype=np.float64)
        t_max = np.empty(N, dtype=np.float64)
        delta_mask = np.zeros(N, dtype=np.bool_)

        for i in prange(N):
            u = long_axis_vectors[i] / a_vals[i]  # 长轴u
            v = short_axis_vectors[i] / b_vals[i]  # 短轴v
            rel = points_on_line[i] - disc_centers[i]

            x0 = u[0] * rel[0] + u[1] * rel[1] + u[2] * rel[2]
            y0 = v[0] * rel[0] + v[1] * rel[1] + v[2] * rel[2]
            dx = u[0] * line_dirs[i, 0] + u[1] * line_dirs[i, 1] + u[2] * line_dirs[i, 2]
            dy = v[0] * line_dirs[i, 0] + v[1] * line_dirs[i, 1] + v[2] * line_dirs[i, 2]

            inv_a = 1.0 / a_vals[i]
            inv_b = 1.0 / b_vals[i]

            x_para1 = dx * inv_a
            y_para1 = dy * inv_b
            x_para2 = x0 * inv_a
            y_para2 = y0 * inv_b

            A = x_para1 ** 2 + y_para1 ** 2
            B = 2.0 * (x_para1 * x_para2 + y_para1 * y_para2)
            C = x_para2 ** 2 + y_para2 ** 2 - 1.0

            Delta = B ** 2 - 4.0 * A * C

            if Delta > 0.0:
                sqrt_D = np.sqrt(Delta)
                t1 = (-B - sqrt_D) / (2.0 * A)
                t2 = (-B + sqrt_D) / (2.0 * A)
                t_min[i] = min(t1, t2)
                t_max[i] = max(t1, t2)
                delta_mask[i] = True
            else:
                t_min[i] = 0.0
                t_max[i] = 0.0
                delta_mask[i] = False

        return t_min, t_max, delta_mask

    # ########################################### __main body of function__ ###########################################
    #  提取结构面参数、交线参数，并构建他们的索引映射
    datalist = []
    disc_to_datalist = {}
    for pair_idx, disc_pair in enumerate(discontinuity_pairs):
        # 判断交线是否共面
        if coplanar_masks[pair_idx]:
            # 说明几乎共面，可以视作无交线或特殊处理
            continue

        # 读取交线参数
        line_dir = line_dirs[pair_idx]  # (shape: (3,))
        point_on_line = points[pair_idx]  # (shape: (3,))

        for disc in disc_pair:
            # 读取椭圆盘参数，以构建椭圆盘局部坐标系
            long_axis_vertex = disc.long_axis_vertex  # point_3d (shape: (3,)) dtype=np.float64
            short_axis_vertex = disc.short_axis_vertex  # point_3d (shape: (3,)) dtype=np.float64
            origin = disc.disc_center  # 局部三维原点 np.asarray([X, Y, Z], dtype=np.float64)
            a = disc.ellip_a  # np.float64
            b = disc.ellip_b  # np.float64

            # pair_idx是line_dirs，discontinuity_pairs，points，coplanar_masks的索引，cluster_id是disc的索引
            data_idx = len(datalist)
            disc_to_datalist[data_idx] = pair_idx

            # 存储数据到datalist
            datalist.append((data_idx,
                             line_dir, point_on_line,
                             long_axis_vertex, short_axis_vertex, origin, a, b))

    # 对datalist中的内容进行numpy向量化，并执行：投影交线到椭圆盘局部二维平面坐标系并求解二维交线段，返回以交线参数方程表达的交线段两端点
    # 注意：先用object数组，在batch_projection_intersect_ellipdisk_2d内分拆字段，方便对datalist分批
    datalist_np = np.array(datalist, dtype=object)

    # 解包datalist_np:拆分各字段
    assert len(datalist) > 0, "datalist 为空"
    line_dirs = np.vstack([row[1] for row in datalist])
    points_on_line = np.vstack([row[2] for row in datalist])
    long_axis_vectors = np.vstack([row[3] for row in datalist])
    short_axis_vectors = np.vstack([row[4] for row in datalist])
    disc_centers = np.vstack([row[5] for row in datalist])
    a_vals = np.array([float(row[6]) for row in datalist], dtype=np.float64)
    b_vals = np.array([float(row[7]) for row in datalist], dtype=np.float64)

    # 执行向量化运算
    t_min, t_max, delta_mask = batch_intersection_2d(disc_centers, long_axis_vectors, short_axis_vectors,
                                                     a_vals, b_vals, points_on_line, line_dirs)

    # 返回与discontinuity_pairs的数据结构对应的intersection_pairs
    intersection_pairs = defaultdict(list)
    delta_false_cont = 0
    for data_idx, delta in enumerate(list(delta_mask)):
        # get pair_idx
        pair_idx = disc_to_datalist[data_idx]

        if not delta:
            delta_false_cont += 1
            intersection_pairs[pair_idx].append((None, None))
        else:
            intersection_pairs[pair_idx].append((t_min[data_idx], t_max[data_idx]))

    # 返回与discontinuity_pairs的数据结构对应的valid_intersection_pairs
    valid_intersection_pairs = {}
    for pair_idx, values in intersection_pairs.items():
        has_None_flag = 0
        for t_pair in values:
            if t_pair == (None, None):
                has_None_flag = 1
        if has_None_flag:
            continue
        valid_intersection_pairs[pair_idx] = values

    return intersection_pairs, valid_intersection_pairs


def extract_seg_from_intsec_pair(intersection_pairs, discontinuity_pairs, line_dirs, points_on_line, tol_relative=1e-6):
    """
    批量矢量化计算交线段
    :param intersection_pairs: Dict[pair_idx(int), List[(t_min, t_max),(t_min, t_max)] 或 [(None, None),(None, None)]
    :param line_dirs: ndarray (N, 3)
    :param points_on_line: ndarray (N, 3)
    :param tol_relative: float，相对误差阈值
    :return: ndarray (M, 2, 3) 有效交线段端点坐标
    """
    datalist = []
    for pair_idx, (disc1, disc2) in enumerate(discontinuity_pairs):
        try:
            intersection1, intersection2 = intersection_pairs[pair_idx]
        except:
            continue
        if intersection1[0] == None or intersection2[0] == None:
            continue
        line_dir = line_dirs[pair_idx]
        point_on_line = points_on_line[pair_idx]
        # datalist 中都是有效数据
        datalist.append([disc1.cluster_id, disc2.cluster_id, pair_idx, line_dir, point_on_line,
                         intersection1[0], intersection1[1], intersection2[0], intersection2[1]])  # 5
    # 提取各列数据
    t1_min = np.array([d[5] for d in datalist])  # shape: (N, )
    t1_max = np.array([d[6] for d in datalist])  # shape: (N, )
    t2_min = np.array([d[7] for d in datalist])  # shape: (N, )
    t2_max = np.array([d[8] for d in datalist])  # shape: (N, )
    used_line_dirs = np.stack([d[3] for d in datalist])  # shape: (N, 3)
    used_points_on_line = np.stack([d[4] for d in datalist])  # shape: (N, 3)
    used_disc_pair_ids = np.stack([d[0:2] for d in datalist])  # shape: (N, 3)

    # 计算一维交集区间
    t_min = np.maximum(t1_min, t2_min)
    t_max = np.minimum(t1_max, t2_max)

    # # 计算容差
    # tol1 = np.abs(t1_max - t1_min)
    # tol2 = np.abs(t2_max - t2_min)
    # tol = tol_relative * np.maximum(tol1, tol2)

    # 防止由于浮点误差导致交线极短的问题,计算容差;考虑交集过短的情况时会把intersection过短的情况也会包含在内
    valid_mask = abs(t_max - t_min) > tol_relative

    # 过滤有效数据
    t_min_valid = t_min[valid_mask]
    t_max_valid = t_max[valid_mask]
    line_dir_valid = used_line_dirs[valid_mask]
    point_on_line_valid = used_points_on_line[valid_mask]
    disc_pair_ids_valid = used_disc_pair_ids[valid_mask]

    # 计算空间端点
    pt_start = point_on_line_valid + t_min_valid[:, np.newaxis] * line_dir_valid
    pt_end = point_on_line_valid + t_max_valid[:, np.newaxis] * line_dir_valid

    # 拼接结果
    segments = []
    for i, coord_start in enumerate(pt_start):
        coord_end = pt_end[i]
        segments.append(
            Segment.Segment(coord_start, coord_end, line_dir_valid[i],
                            (disc_pair_ids_valid[i, 0], disc_pair_ids_valid[i, 1])))
    return segments


def single_extract_seg_from_intsec_pair(args):
    """
    单个结构面对的交线段计算逻辑
    :param args: Tuple，展开为 (disc1, disc2, pair_idx, intersection1, intersection2, line_dir, point_on_line)
    :return: Segment.Segment 或 None
    """
    disc1, disc2, pair_idx, intersection1, intersection2, line_dir, point_on_line = args

    if intersection1[0] is None or intersection2[0] is None:
        return None

    # 考虑交集过短的情况时会把intersection过短的情况也会包含在内
    t1_min, t1_max = intersection1
    t2_min, t2_max = intersection2

    # 求两个区间的交集 [t_min, t_max]
    t_min = max(t1_min, t2_min)
    t_max = min(t1_max, t2_max)

    # 为增强数值稳健性，防止由于浮点误差导致交线极短的问题, 设置为相对误差
    tol = 1e-6 * max(abs(t1_max - t1_min), abs(t2_max - t2_min))
    if t_max - t_min <= tol:
        return None

    # 计算交段的三维空间端点
    pt_start = point_on_line + t_min * line_dir
    pt_end = point_on_line + t_max * line_dir

    return Segment.Segment(pt_start, pt_end, line_dir, (disc1.cluster_id, disc2.cluster_id))


def multiprocess_extract_seg_from_intsec_pair(intersection_pairs, discontinuity_pairs, line_dirs, points, pool_size=16):
    """
    多进程版本，批量计算交线段
    :return: List of valid Segment.Segment
    """

    datalist = []
    for pair_idx, (disc1, disc2) in enumerate(discontinuity_pairs):
        try:
            intersection1, intersection2 = intersection_pairs[pair_idx]
        except KeyError:
            continue
        line_dir = line_dirs[pair_idx]
        point_on_line = points[pair_idx]
        datalist.append((disc1, disc2, pair_idx, intersection1, intersection2, line_dir, point_on_line))

    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.map(single_extract_seg_from_intsec_pair, datalist)

    # 过滤有效结果
    segments = [seg for seg in results if seg is not None]

    # segments = []
    # for pair_idx,(disc1, disc2)  in enumerate(discontinuity_pairs):
    #     # 读取交线参数
    #     line_dir = line_dirs[pair_idx]  # (shape: (3,))
    #     point_on_line = points[pair_idx]  # (shape: (3,))
    #
    #     [intersection1, intersection2] = intersection_pairs[pair_idx]
    #     if intersection1[0] == None or intersection2[0] == None:
    #         continue
    #     else:
    #         # 将两个三维点段 seg1, seg2 映射到 line_dir 上的参数 t 值段
    #         # 直线表示为 p(t) = point_on_line + t * line_dir
    #         t1_min, t1_max = intersection1
    #         t2_min, t2_max = intersection2
    #
    #         t_min = max(t1_min, t2_min)
    #         t_max = min(t1_max, t2_max)
    #
    #         tol = 1e-8 * max(abs(t1_max - t1_min), abs(t2_max - t2_min))
    #         if t_max - t_min <= tol:
    #             continue
    #
    #         pt_start = point_on_line + t_min * line_dir
    #         pt_end = point_on_line + t_max * line_dir
    #
    #         # 存储segment
    #         segments.append(Segment.Segment(pt_start, pt_end, line_dir, (disc1.cluster_id, disc2.cluster_id)))
    return segments


# def multiprocess_discontinuity_intersection(discontinuity_pairs, line_dirs, points, coplanar_masks, pool_size=16):
#     """
#     多进程版本，结合Numba，批量计算结构面交线
#     :param discontinuity_pairs: List[Tuple[Discontinuity, Discontinuity]]
#     :param line_dirs: ndarray, shape (m, 3)，每对结构面的交线方向
#     :param points: ndarray, shape (m, 3)，每条交线上的一点
#     :param coplanar_masks: ndarray, shape (m,)，共面掩码
#     :param pool_size: int，多进程数量
#     :return: List of intersection segments
#     """
#     starttime = time.perf_counter()
#     pool = multiprocessing.Pool(processes=pool_size)
#
#     datalist = list(zip(discontinuity_pairs, line_dirs, points, coplanar_masks))
#     print(f'[report] task number:{len(datalist)}')
#     print(f'[report] pool size:{pool_size}')
#
#     # 全量交给starmap_async，内部自动调度
#     results = pool.starmap_async(segment_ellipdisks, datalist)
#
#     print('Wait for all process pools to finish executing...')
#     segments = results.get()
#     pool.close()
#     pool.join()
#
#     print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — calculation intersections have done.')
#     return segments
#
#
# def multiprocess_discontinuity_intersection_v1(discontinuity_pairs, line_dirs, points, coplanar_masks, pool_size=16):
#     """
#     使用多进程并行计算结构面椭球/椭圆盘之间的交线信息。
#
#     :param discontinuity_pairs: List[Tuple[Discontinuity, Discontinuity]]
#     :param line_dirs: ndarray, shape (m, 3)，每对结构面的交线方向
#     :param points: ndarray, shape (m, 3)，每条交线上的一点
#     :param coplanar_masks: ndarray, shape (m,)，共面掩码
#     :param pool_size: int，多进程数量
#     :return: List of intersection segments
#     """
#     starttime = time.perf_counter()
#     pool = multiprocessing.Pool(processes=pool_size)
#
#     # 构造组合数据：每一项是一个 tuple 包含所需数据
#     datalist = list(zip(discontinuity_pairs, line_dirs, points, coplanar_masks))
#
#     # 使用 list 切分方式（替代 np.array_split）将datalist切成poolsize份
#     def chunk_list(datalist, n_chunks):
#         chunk_size = math.ceil(len(datalist) / n_chunks)
#         return [datalist[i:i + chunk_size] for i in range(0, len(datalist), chunk_size)]
#
#     batches = chunk_list(datalist, pool_size)
#
#     print('pool_size:', pool_size)
#     print('batch_size:', len(batches[0]))
#     print('batch_number:', len(batches))
#
#     # For each batch, a process pool is started
#     results = []
#     for batch in batches:
#         result = pool.starmap_async(segment_ellipdisks,
#                                     [tuple_data for tuple_data in batch],
#                                     error_callback=print_error)
#         results.append(result)
#
#     # Wait for all process pools to finish executing
#     print('Waiting for all subprocesses done...')
#     for result in results:
#         result.wait()
#
#     # 获取所有子进程返回的结果（二维列表），再拉平成一维
#     true_results = [r.get() for r in results]  # => List of List
#     segments = [item for sublist in true_results for item in sublist]
#
#     # Closing the process pool
#     pool.close()
#     pool.join()
#     print(
#         f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — calculate intersections of discontinuity disks.')
#     return segments


# def segment_ellipdisks(tuple_data):
#     """
#     计算两个椭圆盘结构面之间的交线与两个椭圆盘的交点（即线段两端点）
#
#     :param tuple_data: ((disc1, disc2), line_dir, point, is_coplanar)
#     :return: Segment.Segment(pt1, pt2,surface_ids=(id1, id2)) 若无交点则为 [None, None]
#     """
#
#     @njit
#     def intersect_line_with_ellipdisc(disc, point_on_line, line_dir):
#         """
#         计算一条空间直线与椭圆盘结构面的交点（最多2个）
#
#         :param disc: 椭圆盘结构面对象，需具有 projections_uv、centroid、ellip_a、ellip_b 属性
#         :param point_on_line: 空间直线上的一点 (3,)
#         :param line_dir: 空间直线的方向向量 (3,)
#         :return: None 或 (pt1, pt2) 两个交点
#         """
#         # 读取椭圆盘数据，并以椭圆盘构建局部坐标系
#         long_axis_vertex = disc.long_axis_vertex
#         short_axis_vertex = disc.short_axis_vertex
#         origin = disc.disc_center  # 局部三维原点
#         a = disc.ellip_a
#         b = disc.ellip_b
#         u = long_axis_vertex / a  # 长轴u 三维单位向量
#         v = short_axis_vertex / b  # 短轴v 三维单位向量
#
#         # 直线映射到椭圆盘所在二维平面
#         rel = point_on_line - origin
#         x0 = np.dot(rel, u)  # 将 rel 向量分别投影到 u、v 基底上，得到该点在椭圆盘局部坐标系中的二维坐标 (x0, y0)
#         y0 = np.dot(rel, v)
#         dx = np.dot(line_dir, u)  # 将直线方向向量 line_dir 投影到 u 和 v 方向上。得到直线在椭圆盘二维平面中的方向向量 (dx, dy)
#         dy = np.dot(line_dir, v)
#
#         # 求解椭圆方程下的二次方程：A t^2 + B t + C = 0
#         x_para1, y_para1 = dx / a, dy / b
#         x_para2, y_para2 = x0 / a, y0 / b
#         A = x_para1 * x_para1 + y_para1 * y_para1
#         B = 2 * (x_para1 * x_para2 + y_para1 * y_para2)
#         C = x_para2 * x_para2 + y_para2 * y_para2
#
#         Delta = B ** 2 - 4 * A * C
#         if Delta <= 0:
#             return None  # 无交点或只有1个交点
#
#         sqrt_D = np.sqrt(Delta)
#         t1 = (-B - sqrt_D) / (2 * A)
#         t2 = (-B + sqrt_D) / (2 * A)
#         # pt1 = point_on_line + t1 * line_dir
#         # pt2 = point_on_line + t2 * line_dir
#         return (min(t1, t2), max(t1, t2))  # 保证顺序：小值在前
#
#     # ########################################### __main body of function__ ###########################################
#     # 交线（空间直线）的方向向量
#     pt_none = [0, 0, 0]
#     (disc1, disc2), line_dir, point_on_line, is_coplanar = tuple_data
#
#     # 判断交线是否共面
#     if is_coplanar:
#         # 说明几乎共面，可以视作无交线或特殊处理
#         return Segment.Segment(pt_none, pt_none, line_dir, None)
#
#     # 将line_dir分别投影到disc1, disc2上计算截线段
#     seg1_t = intersect_line_with_ellipdisc(disc1, point_on_line, line_dir)
#     seg2_t = intersect_line_with_ellipdisc(disc2, point_on_line, line_dir)
#
#     if seg1_t is None or seg1_t is None:
#         return Segment.Segment(pt_none, pt_none, line_dir, None)
#
#     # 将两个三维点段 seg1, seg2 映射到 line_dir 上的参数 t 值段
#     # 直线表示为 p(t) = point_on_line + t * line_dir
#     t1_min, t1_max = seg1_t
#     t2_min, t2_max = seg2_t
#
#     # 求两个区间的交集 [t_min, t_max]
#     t_min = max(t1_min, t2_min)
#     t_max = min(t1_max, t2_max)
#
#     # 为增强数值稳健性，防止由于浮点误差导致交线极短的问题, 设置为相对误差
#     tol = 1e-8 * max(abs(t1_max - t1_min), abs(t2_max - t2_min))
#     if t_max - t_min <= tol:
#         return Segment.Segment(pt_none, pt_none, line_dir, None)  # 无交段
#
#     # 计算交段的三维空间端点
#     pt_start = point_on_line + t_min * line_dir
#     pt_end = point_on_line + t_max * line_dir
#
#     return Segment.Segment(pt_start, pt_end, line_dir, (disc1.cluster_id, disc2.cluster_id))


def intersect_plane(discontinuity_pairs, coplanar_tol=1e-6):
    """
    批量计算多个结构面对之间的交线方向、交点及共面性判断。

    :param discontinuity_pairs: List[Tuple[Discontinuity, Discontinuity]]
    :param coplanar_tol: float, 判断是否共面的阈值
    :return:
        line_dirs: (m, 3) ndarray，每对平面的交线方向（若共面为0向量）
        points_on_lines: (m, 3) ndarray，交线上的一点（共面则为np.nan）
        coplanar_mask: (m,) ndarray，bool数组，表示是否共面
    """
    if len(discontinuity_pairs) > 10_000:
        print(f"[警告] 待计算的结构面组合数超过1万对：{len(discontinuity_pairs)}，可能造成内存爆炸。")

    n1s = np.asarray([pair[0].normal for pair in discontinuity_pairs], dtype=np.float64)  # (m, 3)
    n2s = np.asarray([pair[1].normal for pair in discontinuity_pairs], dtype=np.float64)  # (m, 3)
    p1s = np.asarray([pair[0].disc_center for pair in discontinuity_pairs], dtype=np.float64)  # (m, 3)
    p2s = np.asarray([pair[1].disc_center for pair in discontinuity_pairs], dtype=np.float64)  # (m, 3)
    assert n1s.shape == (len(discontinuity_pairs), 3)
    assert n2s.shape == (len(discontinuity_pairs), 3)
    assert p1s.shape == (len(discontinuity_pairs), 3)
    assert p2s.shape == (len(discontinuity_pairs), 3)

    # 交线方向向量（未归一化）
    raw_dirs = np.cross(n1s, n2s)  # (m, 3)
    assert raw_dirs.shape == (len(discontinuity_pairs), 3)
    norms = np.linalg.norm(raw_dirs, axis=1)  # (m)
    assert len(norms.shape) == 1

    # 共面检测：交线方向模长是否接近0
    coplanar_mask = norms < coplanar_tol
    line_dirs = np.zeros_like(raw_dirs)
    line_dirs[~coplanar_mask] = raw_dirs[~coplanar_mask] / norms[~coplanar_mask, None]  # 归一化后的方向
    assert raw_dirs.shape[1] == 3

    # 平面方程的 d = n·p
    d1s = np.einsum('ij,ij->i', n1s, p1s)
    d2s = np.einsum('ij,ij->i', n2s, p2s)

    # 初始化交点数组
    points_on_lines = np.full_like(n1s, np.nan)

    for i in range(len(discontinuity_pairs)):
        if coplanar_mask[i]:
            continue
        A = np.column_stack([n1s[i], n2s[i], line_dirs[i]])  # (3, 3)
        b = np.array([d1s[i], d2s[i], 0.0])
        try:
            x = np.linalg.solve(A, b)
            points_on_lines[i] = x
        except np.linalg.LinAlgError:
            # 奇异矩阵时处理为 nan
            coplanar_mask[i] = True
            points_on_lines[i] = np.nan
            line_dirs[i] = 0.0

    return line_dirs, points_on_lines, coplanar_mask


# def intersect_segments(segments, tol=1e-6):
#

# def intersect_segments_testv1(seg_a: Segment.Segment, seg_b: Segment.Segment, tol=1e-6):
#     """
#     判断两条 Segment 对象是否相交，支持容差，返回交点 np.ndarray(3,) 或 None。
#     """
#     # 判断是否为空
#     if seg_a is None or seg_b is None:
#         return None
#
#     # CGAL 表示
#     try:
#         s1 = Segment_3(Point_3(*seg_a.p1), Point_3(*seg_a.p2))
#         s2 = Segment_3(Point_3(*seg_b.p1), Point_3(*seg_b.p2))
#         obj = CGAL_Kernel.intersection(s1, s2)
#     except Exception:
#         return None
#
#     # 若无交点
#     if obj is None:
#         return None
#
#     # 返回交点（Point_3）
#     try:
#         pt = obj.get_Point_3()
#         return np.array([pt.x(), pt.y(), pt.z()], dtype=np.float64)
#     except:
#         # 若不是点交，可能是重合 segment，此时视为无交点
#         return None
#
#
# def parallel_segment_intersections_testv1(segments, tol=1e-6, pool_size=8):
#     def get_segment_aabbs(segments):
#         """
#         segments: List[Segment]
#         返回各 segment 的 [xmin, ymin, zmin, xmax, ymax, zmax]
#         """
#         boxes = []
#         for seg in segments:
#             p1, p2 = seg.p1, seg.p2
#             box = np.hstack([np.minimum(p1, p2), np.maximum(p1, p2)])
#             boxes.append(box)
#         return np.array(boxes)
#
#     def aabb_intersect(a, b):
#         return np.all(a[:3] <= b[3:]) and np.all(b[:3] <= a[3:])
#
#     def task(args):
#         i, j = args
#         pt = intersect_segments_testv1(segments[i], segments[j], tol)
#         if pt is not None:
#             return (i, j, pt)
#         else:
#             return None
#
#     # ########################################### __main body of function__ ###########################################
#     boxes = get_segment_aabbs(segments)
#     candidate_pairs = [(i, j) for i, j in combinations(range(len(segments)), 2)
#                        if aabb_intersect(boxes[i], boxes[j])]
#     with Pool(pool_size) as pool:
#         results = pool.map(task, candidate_pairs)
#
#     return [r for r in results if r is not None]  # 结果格式: [(i, j, pt),...]


def aggregate_nodes_with_tolerance(nodes, segments, tol=1e-6):
    """
    聚合交点：将距离小于 tol 的交点聚为同一个点。

    输入：
        nodes: List of (segid_i, segid_j, pt) 三元组，pt 是 ndarray(3,)
        segments: 原始 Segment 列表，用于回查 surface_id 信息
        tol: 距离容差

    返回：
        unodes_coord: ndarray(N, 3)，唯一聚合点坐标
        unodes_by_disc: Dict[surface_id, List[(seg_idx_i, seg_idx_j, unod_idx)]]
    """
    nodes_coord = np.array([pt for _, _, pt in nodes])
    tree = KDTree(nodes_coord)

    # 获取每个点在 tol 半径内的所有点索引
    neighbors = tree.query_ball_point(nodes_coord, r=tol)

    # 使用并查集构造点的聚合组
    # 并查集（Union-Find）结构：用于将多个点根据距离容差（如空间点之间的距离小于 tol）聚合为若干个集合。
    # 它的作用是：高效地判断哪些点属于同一个“点簇”或“点团”，并支持动态合并
    parent = np.arange(len(nodes_coord))  # 生成一个整数数组，从 0 到 len(nodes_coord)-1

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    for i, group in enumerate(neighbors):
        for j in group:
            union(i, j)

    # 生成最终唯一点及映射
    cluster_map = {}
    unodes_coord = []  # 记录唯一点upt的坐标值
    node_unode_indices = []  # 以node的索引 查询唯一点upt的索引

    for k in range(len(nodes_coord)):
        root = find(k)
        if root not in cluster_map:
            cluster_map[root] = len(unodes_coord)
            unodes_coord.append(nodes_coord[root])
        node_unode_indices.append(cluster_map[root])

    # 构建边图：按结构面组织
    unodes_by_disc = defaultdict(list)
    for (i, j, _), idx in zip(nodes, node_unode_indices):
        # i,j is segments index
        # idx is unode point index
        seg_i = segments[i]
        seg_j = segments[j]
        common_surfaces = set(seg_i.surface_ids) & set(seg_j.surface_ids)
        for cluster_id in common_surfaces:
            # 存储以cluster_id的列表字典，每个列表中包含了node对应的唯一点索引[为什么只记录1个点？没有edge信息啊，不是所有点都相互连接]
            unodes_by_disc[cluster_id].append((i, j, idx))  # (seg_i_idx, seg_j_idx, upt_idx)

    return np.array(unodes_coord), unodes_by_disc


def extract_surface_loops(unodes_coord, unodes_by_disc, segments, pool_size=None):
    '''

    :param segments:List[Segment]
    :param nodes:[(segment_i, segment_j, pt),...]   pt是array[pt.x(), pt.y(), pt.z()]
    :param tol:
    :param pool_size:
    :return:
        point_coords: 所有唯一聚合交点 np.ndarray 列表
        surface_loops: Dict[surface_id → List[Loop]]，每个 loop 是点的坐标列表
            loops = [[unodes_coord[idx] for idx in cycle] for cycle in cycles]
    '''

    def build_graph_and_find_cycles(args):
        # cluster_id, unodes_idx_list, segments, unodes_coord
        cluster_id, unodes_idx_list, segments, unodes_coord = args
        g = nx.Graph()
        seg_unodes = defaultdict(list)  # segment_id -> list of (idx, coord)

        # Step 1: 将 unodes 按 segment 聚集
        for seg_i, seg_j, idx in unodes_idx_list:
            for s in [seg_i, seg_j]:
                seg_unodes[s].append((idx, unodes_coord[idx]))

        # Step 2: 对每条 segment 上的点排序并连边
        for seg_id, pts in seg_unodes.items():
            if len(pts) < 2:
                continue
            # 获取当前 segment 的方向
            seg = segments[seg_id]
            start, end = seg.p1, seg.p2  # 假设 segment 有 start/end 属性
            dir_vec = seg.dir

            # 按投影距离排序
            pts.sort(key=lambda x: np.dot(x[1] - start, dir_vec))  # (idx, coord)

            # 连续两两相连
            for (idx1, _), (idx2, _) in zip(pts[:-1], pts[1:]):
                g.add_edge(idx1, idx2)

        # Step 3: 提取闭环
        cycles = nx.cycle_basis(g)
        loops = [[unodes_coord[idx] for idx in cycle] for cycle in cycles]
        return cluster_id, loops

    def build_graph_and_find_cycles_test1(args):
        cluster_id, unodes_idx_list, segments, point_coords = args
        g = nx.Graph()
        for i, j, unode_idx in unodes_idx_list:
            seg_i = segments[i]
            seg_j = segments[j]
            # 得到两个 segment 中关于该 surface 的边段（交点分布在 segment 上）
            for seg in [seg_i, seg_j]:
                if cluster_id not in seg.surface_ids:
                    continue
                # 连接 segment 上该 surface 的交点段
                # 收集disc上其他与seg_i或者seg_j有关的unode
                other_pts = []
                for node in unodes_idx_list:
                    if node[2] != unode_idx and (node[0] == i or node[1] == i or node[0] == j or node[1] == j):
                        other_pts.append(node[2])
                # 将给定点unode_idx与其他点之间的边加到图中
                '''
                问题出在这里，以某个节点为初始点，连接任何在其2条seg上的所有其他node会导致同一条seg上的node不是顺序连接，而是相互连接
                '''
                for other in other_pts:
                    g.add_edge(unode_idx, other)

        # 提取闭环（cycle basis）
        cycles = nx.cycle_basis(g)
        loops = [[point_coords[idx] for idx in cycle] for cycle in cycles]
        return cluster_id, loops

    # ########################################### __main body of function__ ###########################################
    if pool_size is None:
        pool_size = max(cpu_count() - 1, 1)

    args_list = [
        (cluster_id, unodes_idx_list, segments, unodes_coord)
        for cluster_id, unodes_idx_list in unodes_by_disc.items()
    ]

    with Pool(processes=pool_size) as pool:
        results = pool.map(build_graph_and_find_cycles, args_list)

    surface_loops = dict(results)
    return surface_loops


class HalfFace:
    """
    半面（Half-Face）数据结构。

    属性：
        cluster_id: int，所属结构面ID。
        pt_indices: Tuple[int]，该面的顶点索引列表。
        normal: Tuple[float]，该面的法向量。
    """

    def __init__(self, cluster_id, pt_indices, normal):
        self.cluster_id = cluster_id
        self.pt_indices = tuple(pt_indices)
        self.normal = tuple(normal)
        self.id = None

    def edges(self):
        """返回该面所有边的标准化形式（两端点索引排序）。"""
        return [tuple(sorted((self.pt_indices[i], self.pt_indices[(i + 1) % len(self.pt_indices)]))) for i in
                range(len(self.pt_indices))]

    def index(self, hf_idx):
        self.id = hf_idx

    def __hash__(self):
        return hash((self.cluster_id, self.pt_indices, self.normal))

    def __eq__(self, other):
        return (self.cluster_id, self.pt_indices, self.normal) == (other.cluster_id, other.pt_indices, other.normal)


####################################################################
def print_error(value):
    '''
    这个函数可以输出多进程中的报错，但是不会终止多进程
    '''
    print("error: ", value)


def time_cost_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} h {m} min {s:.2f} sec"


################################## rubbish
def single_extract_seg_from_intsec_pair(args):
    """
    单个结构面对的交线段计算逻辑
    :param args: Tuple，展开为 (disc1, disc2, pair_idx, intersection1, intersection2, line_dir, point_on_line)
    :return: Segment.Segment 或 None
    """
    disc1, disc2, pair_idx, intersection1, intersection2, line_dir, point_on_line = args

    if intersection1[0] is None or intersection2[0] is None:
        return None

    # 考虑交集过短的情况时会把intersection过短的情况也会包含在内
    t1_min, t1_max = intersection1
    t2_min, t2_max = intersection2

    # 求两个区间的交集 [t_min, t_max]
    t_min = max(t1_min, t2_min)
    t_max = min(t1_max, t2_max)

    # 为增强数值稳健性，防止由于浮点误差导致交线极短的问题, 设置为相对误差
    tol = 1e-6 * max(abs(t1_max - t1_min), abs(t2_max - t2_min))
    if t_max - t_min <= tol:
        return None

    # 计算交段的三维空间端点
    pt_start = point_on_line + t_min * line_dir
    pt_end = point_on_line + t_max * line_dir

    return Segment.Segment(pt_start, pt_end, line_dir, (disc1.cluster_id, disc2.cluster_id))


def multiprocess_extract_seg_from_intsec_pair(intersection_pairs, discontinuity_pairs, line_dirs, points, pool_size=16):
    """
    多进程版本，批量计算交线段
    :return: List of valid Segment.Segment
    """

    datalist = []
    for pair_idx, (disc1, disc2) in enumerate(discontinuity_pairs):
        try:
            intersection1, intersection2 = intersection_pairs[pair_idx]
        except KeyError:
            continue
        line_dir = line_dirs[pair_idx]
        point_on_line = points[pair_idx]
        datalist.append((disc1, disc2, pair_idx, intersection1, intersection2, line_dir, point_on_line))

    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.map(single_extract_seg_from_intsec_pair, datalist)

    # 过滤有效结果
    segments = [seg for seg in results if seg is not None]

    # segments = []
    # for pair_idx,(disc1, disc2)  in enumerate(discontinuity_pairs):
    #     # 读取交线参数
    #     line_dir = line_dirs[pair_idx]  # (shape: (3,))
    #     point_on_line = points[pair_idx]  # (shape: (3,))
    #
    #     [intersection1, intersection2] = intersection_pairs[pair_idx]
    #     if intersection1[0] == None or intersection2[0] == None:
    #         continue
    #     else:
    #         # 将两个三维点段 seg1, seg2 映射到 line_dir 上的参数 t 值段
    #         # 直线表示为 p(t) = point_on_line + t * line_dir
    #         t1_min, t1_max = intersection1
    #         t2_min, t2_max = intersection2
    #
    #         t_min = max(t1_min, t2_min)
    #         t_max = min(t1_max, t2_max)
    #
    #         tol = 1e-8 * max(abs(t1_max - t1_min), abs(t2_max - t2_min))
    #         if t_max - t_min <= tol:
    #             continue
    #
    #         pt_start = point_on_line + t_min * line_dir
    #         pt_end = point_on_line + t_max * line_dir
    #
    #         # 存储segment
    #         segments.append(Segment.Segment(pt_start, pt_end, line_dir, (disc1.cluster_id, disc2.cluster_id)))
    return segments


# def multiprocess_discontinuity_intersection(discontinuity_pairs, line_dirs, points, coplanar_masks, pool_size=16):
#     """
#     多进程版本，结合Numba，批量计算结构面交线
#     :param discontinuity_pairs: List[Tuple[Discontinuity, Discontinuity]]
#     :param line_dirs: ndarray, shape (m, 3)，每对结构面的交线方向
#     :param points: ndarray, shape (m, 3)，每条交线上的一点
#     :param coplanar_masks: ndarray, shape (m,)，共面掩码
#     :param pool_size: int，多进程数量
#     :return: List of intersection segments
#     """
#     starttime = time.perf_counter()
#     pool = multiprocessing.Pool(processes=pool_size)
#
#     datalist = list(zip(discontinuity_pairs, line_dirs, points, coplanar_masks))
#     print(f'[report] task number:{len(datalist)}')
#     print(f'[report] pool size:{pool_size}')
#
#     # 全量交给starmap_async，内部自动调度
#     results = pool.starmap_async(segment_ellipdisks, datalist)
#
#     print('Wait for all process pools to finish executing...')
#     segments = results.get()
#     pool.close()
#     pool.join()
#
#     print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — calculation intersections have done.')
#     return segments
#
#
# def multiprocess_discontinuity_intersection_v1(discontinuity_pairs, line_dirs, points, coplanar_masks, pool_size=16):
#     """
#     使用多进程并行计算结构面椭球/椭圆盘之间的交线信息。
#
#     :param discontinuity_pairs: List[Tuple[Discontinuity, Discontinuity]]
#     :param line_dirs: ndarray, shape (m, 3)，每对结构面的交线方向
#     :param points: ndarray, shape (m, 3)，每条交线上的一点
#     :param coplanar_masks: ndarray, shape (m,)，共面掩码
#     :param pool_size: int，多进程数量
#     :return: List of intersection segments
#     """
#     starttime = time.perf_counter()
#     pool = multiprocessing.Pool(processes=pool_size)
#
#     # 构造组合数据：每一项是一个 tuple 包含所需数据
#     datalist = list(zip(discontinuity_pairs, line_dirs, points, coplanar_masks))
#
#     # 使用 list 切分方式（替代 np.array_split）将datalist切成poolsize份
#     def chunk_list(datalist, n_chunks):
#         chunk_size = math.ceil(len(datalist) / n_chunks)
#         return [datalist[i:i + chunk_size] for i in range(0, len(datalist), chunk_size)]
#
#     batches = chunk_list(datalist, pool_size)
#
#     print('pool_size:', pool_size)
#     print('batch_size:', len(batches[0]))
#     print('batch_number:', len(batches))
#
#     # For each batch, a process pool is started
#     results = []
#     for batch in batches:
#         result = pool.starmap_async(segment_ellipdisks,
#                                     [tuple_data for tuple_data in batch],
#                                     error_callback=print_error)
#         results.append(result)
#
#     # Wait for all process pools to finish executing
#     print('Waiting for all subprocesses done...')
#     for result in results:
#         result.wait()
#
#     # 获取所有子进程返回的结果（二维列表），再拉平成一维
#     true_results = [r.get() for r in results]  # => List of List
#     segments = [item for sublist in true_results for item in sublist]
#
#     # Closing the process pool
#     pool.close()
#     pool.join()
#     print(
#         f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — calculate intersections of discontinuity disks.')
#     return segments


# def segment_ellipdisks(tuple_data):
#     """
#     计算两个椭圆盘结构面之间的交线与两个椭圆盘的交点（即线段两端点）
#
#     :param tuple_data: ((disc1, disc2), line_dir, point, is_coplanar)
#     :return: Segment.Segment(pt1, pt2,surface_ids=(id1, id2)) 若无交点则为 [None, None]
#     """
#
#     @njit
#     def intersect_line_with_ellipdisc(disc, point_on_line, line_dir):
#         """
#         计算一条空间直线与椭圆盘结构面的交点（最多2个）
#
#         :param disc: 椭圆盘结构面对象，需具有 projections_uv、centroid、ellip_a、ellip_b 属性
#         :param point_on_line: 空间直线上的一点 (3,)
#         :param line_dir: 空间直线的方向向量 (3,)
#         :return: None 或 (pt1, pt2) 两个交点
#         """
#         # 读取椭圆盘数据，并以椭圆盘构建局部坐标系
#         long_axis_vertex = disc.long_axis_vertex
#         short_axis_vertex = disc.short_axis_vertex
#         origin = disc.disc_center  # 局部三维原点
#         a = disc.ellip_a
#         b = disc.ellip_b
#         u = long_axis_vertex / a  # 长轴u 三维单位向量
#         v = short_axis_vertex / b  # 短轴v 三维单位向量
#
#         # 直线映射到椭圆盘所在二维平面
#         rel = point_on_line - origin
#         x0 = np.dot(rel, u)  # 将 rel 向量分别投影到 u、v 基底上，得到该点在椭圆盘局部坐标系中的二维坐标 (x0, y0)
#         y0 = np.dot(rel, v)
#         dx = np.dot(line_dir, u)  # 将直线方向向量 line_dir 投影到 u 和 v 方向上。得到直线在椭圆盘二维平面中的方向向量 (dx, dy)
#         dy = np.dot(line_dir, v)
#
#         # 求解椭圆方程下的二次方程：A t^2 + B t + C = 0
#         x_para1, y_para1 = dx / a, dy / b
#         x_para2, y_para2 = x0 / a, y0 / b
#         A = x_para1 * x_para1 + y_para1 * y_para1
#         B = 2 * (x_para1 * x_para2 + y_para1 * y_para2)
#         C = x_para2 * x_para2 + y_para2 * y_para2
#
#         Delta = B ** 2 - 4 * A * C
#         if Delta <= 0:
#             return None  # 无交点或只有1个交点
#
#         sqrt_D = np.sqrt(Delta)
#         t1 = (-B - sqrt_D) / (2 * A)
#         t2 = (-B + sqrt_D) / (2 * A)
#         # pt1 = point_on_line + t1 * line_dir
#         # pt2 = point_on_line + t2 * line_dir
#         return (min(t1, t2), max(t1, t2))  # 保证顺序：小值在前
#
#     # ########################################### __main body of function__ ###########################################
#     # 交线（空间直线）的方向向量
#     pt_none = [0, 0, 0]
#     (disc1, disc2), line_dir, point_on_line, is_coplanar = tuple_data
#
#     # 判断交线是否共面
#     if is_coplanar:
#         # 说明几乎共面，可以视作无交线或特殊处理
#         return Segment.Segment(pt_none, pt_none, line_dir, None)
#
#     # 将line_dir分别投影到disc1, disc2上计算截线段
#     seg1_t = intersect_line_with_ellipdisc(disc1, point_on_line, line_dir)
#     seg2_t = intersect_line_with_ellipdisc(disc2, point_on_line, line_dir)
#
#     if seg1_t is None or seg1_t is None:
#         return Segment.Segment(pt_none, pt_none, line_dir, None)
#
#     # 将两个三维点段 seg1, seg2 映射到 line_dir 上的参数 t 值段
#     # 直线表示为 p(t) = point_on_line + t * line_dir
#     t1_min, t1_max = seg1_t
#     t2_min, t2_max = seg2_t
#
#     # 求两个区间的交集 [t_min, t_max]
#     t_min = max(t1_min, t2_min)
#     t_max = min(t1_max, t2_max)
#
#     # 为增强数值稳健性，防止由于浮点误差导致交线极短的问题, 设置为相对误差
#     tol = 1e-8 * max(abs(t1_max - t1_min), abs(t2_max - t2_min))
#     if t_max - t_min <= tol:
#         return Segment.Segment(pt_none, pt_none, line_dir, None)  # 无交段
#
#     # 计算交段的三维空间端点
#     pt_start = point_on_line + t_min * line_dir
#     pt_end = point_on_line + t_max * line_dir
#
#     return Segment.Segment(pt_start, pt_end, line_dir, (disc1.cluster_id, disc2.cluster_id))


# def intersect_segments(segments, tol=1e-6):
#

# def intersect_segments_testv1(seg_a: Segment.Segment, seg_b: Segment.Segment, tol=1e-6):
#     """
#     判断两条 Segment 对象是否相交，支持容差，返回交点 np.ndarray(3,) 或 None。
#     """
#     # 判断是否为空
#     if seg_a is None or seg_b is None:
#         return None
#
#     # CGAL 表示
#     try:
#         s1 = Segment_3(Point_3(*seg_a.p1), Point_3(*seg_a.p2))
#         s2 = Segment_3(Point_3(*seg_b.p1), Point_3(*seg_b.p2))
#         obj = CGAL_Kernel.intersection(s1, s2)
#     except Exception:
#         return None
#
#     # 若无交点
#     if obj is None:
#         return None
#
#     # 返回交点（Point_3）
#     try:
#         pt = obj.get_Point_3()
#         return np.array([pt.x(), pt.y(), pt.z()], dtype=np.float64)
#     except:
#         # 若不是点交，可能是重合 segment，此时视为无交点
#         return None
#
#
# def parallel_segment_intersections_testv1(segments, tol=1e-6, pool_size=8):
#     def get_segment_aabbs(segments):
#         """
#         segments: List[Segment]
#         返回各 segment 的 [xmin, ymin, zmin, xmax, ymax, zmax]
#         """
#         boxes = []
#         for seg in segments:
#             p1, p2 = seg.p1, seg.p2
#             box = np.hstack([np.minimum(p1, p2), np.maximum(p1, p2)])
#             boxes.append(box)
#         return np.array(boxes)
#
#     def aabb_intersect(a, b):
#         return np.all(a[:3] <= b[3:]) and np.all(b[:3] <= a[3:])
#
#     def task(args):
#         i, j = args
#         pt = intersect_segments_testv1(segments[i], segments[j], tol)
#         if pt is not None:
#             return (i, j, pt)
#         else:
#             return None
#
#     # ########################################### __main body of function__ ###########################################
#     boxes = get_segment_aabbs(segments)
#     candidate_pairs = [(i, j) for i, j in combinations(range(len(segments)), 2)
#                        if aabb_intersect(boxes[i], boxes[j])]
#     with Pool(pool_size) as pool:
#         results = pool.map(task, candidate_pairs)
#
#     return [r for r in results if r is not None]  # 结果格式: [(i, j, pt),...]


def aggregate_nodes_with_tolerance(nodes, segments, tol=1e-6):
    """
    聚合交点：将距离小于 tol 的交点聚为同一个点。

    输入：
        nodes: List of (segid_i, segid_j, pt) 三元组，pt 是 ndarray(3,)
        segments: 原始 Segment 列表，用于回查 surface_id 信息
        tol: 距离容差

    返回：
        unodes_coord: ndarray(N, 3)，唯一聚合点坐标
        unodes_by_disc: Dict[surface_id, List[(seg_idx_i, seg_idx_j, unod_idx)]]
    """
    nodes_coord = np.array([pt for _, _, pt in nodes])
    tree = KDTree(nodes_coord)

    # 获取每个点在 tol 半径内的所有点索引
    neighbors = tree.query_ball_point(nodes_coord, r=tol)

    # 使用并查集构造点的聚合组
    # 并查集（Union-Find）结构：用于将多个点根据距离容差（如空间点之间的距离小于 tol）聚合为若干个集合。
    # 它的作用是：高效地判断哪些点属于同一个“点簇”或“点团”，并支持动态合并
    parent = np.arange(len(nodes_coord))  # 生成一个整数数组，从 0 到 len(nodes_coord)-1

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    for i, group in enumerate(neighbors):
        for j in group:
            union(i, j)

    # 生成最终唯一点及映射
    cluster_map = {}
    unodes_coord = []  # 记录唯一点upt的坐标值
    node_unode_indices = []  # 以node的索引 查询唯一点upt的索引

    for k in range(len(nodes_coord)):
        root = find(k)
        if root not in cluster_map:
            cluster_map[root] = len(unodes_coord)
            unodes_coord.append(nodes_coord[root])
        node_unode_indices.append(cluster_map[root])

    # 构建边图：按结构面组织
    unodes_by_disc = defaultdict(list)
    for (i, j, _), idx in zip(nodes, node_unode_indices):
        # i,j is segments index
        # idx is unode point index
        seg_i = segments[i]
        seg_j = segments[j]
        common_surfaces = set(seg_i.surface_ids) & set(seg_j.surface_ids)
        for cluster_id in common_surfaces:
            # 存储以cluster_id的列表字典，每个列表中包含了node对应的唯一点索引[为什么只记录1个点？没有edge信息啊，不是所有点都相互连接]
            unodes_by_disc[cluster_id].append((i, j, idx))  # (seg_i_idx, seg_j_idx, upt_idx)

    return np.array(unodes_coord), unodes_by_disc


def extract_surface_loops(unodes_coord, unodes_by_disc, segments, pool_size=None):
    '''

    :param segments:List[Segment]
    :param nodes:[(segment_i, segment_j, pt),...]   pt是array[pt.x(), pt.y(), pt.z()]
    :param tol:
    :param pool_size:
    :return:
        point_coords: 所有唯一聚合交点 np.ndarray 列表
        surface_loops: Dict[surface_id → List[Loop]]，每个 loop 是点的坐标列表
            loops = [[unodes_coord[idx] for idx in cycle] for cycle in cycles]
    '''

    def build_graph_and_find_cycles(args):
        # cluster_id, unodes_idx_list, segments, unodes_coord
        cluster_id, unodes_idx_list, segments, unodes_coord = args
        g = nx.Graph()
        seg_unodes = defaultdict(list)  # segment_id -> list of (idx, coord)

        # Step 1: 将 unodes 按 segment 聚集
        for seg_i, seg_j, idx in unodes_idx_list:
            for s in [seg_i, seg_j]:
                seg_unodes[s].append((idx, unodes_coord[idx]))

        # Step 2: 对每条 segment 上的点排序并连边
        for seg_id, pts in seg_unodes.items():
            if len(pts) < 2:
                continue
            # 获取当前 segment 的方向
            seg = segments[seg_id]
            start, end = seg.p1, seg.p2  # 假设 segment 有 start/end 属性
            dir_vec = seg.dir

            # 按投影距离排序
            pts.sort(key=lambda x: np.dot(x[1] - start, dir_vec))  # (idx, coord)

            # 连续两两相连
            for (idx1, _), (idx2, _) in zip(pts[:-1], pts[1:]):
                g.add_edge(idx1, idx2)

        # Step 3: 提取闭环
        cycles = nx.cycle_basis(g)
        loops = [[unodes_coord[idx] for idx in cycle] for cycle in cycles]
        return cluster_id, loops

    def build_graph_and_find_cycles_test1(args):
        cluster_id, unodes_idx_list, segments, point_coords = args
        g = nx.Graph()
        for i, j, unode_idx in unodes_idx_list:
            seg_i = segments[i]
            seg_j = segments[j]
            # 得到两个 segment 中关于该 surface 的边段（交点分布在 segment 上）
            for seg in [seg_i, seg_j]:
                if cluster_id not in seg.surface_ids:
                    continue
                # 连接 segment 上该 surface 的交点段
                # 收集disc上其他与seg_i或者seg_j有关的unode
                other_pts = []
                for node in unodes_idx_list:
                    if node[2] != unode_idx and (node[0] == i or node[1] == i or node[0] == j or node[1] == j):
                        other_pts.append(node[2])
                # 将给定点unode_idx与其他点之间的边加到图中
                '''
                问题出在这里，以某个节点为初始点，连接任何在其2条seg上的所有其他node会导致同一条seg上的node不是顺序连接，而是相互连接
                '''
                for other in other_pts:
                    g.add_edge(unode_idx, other)

        # 提取闭环（cycle basis）
        cycles = nx.cycle_basis(g)
        loops = [[point_coords[idx] for idx in cycle] for cycle in cycles]
        return cluster_id, loops

    # ########################################### __main body of function__ ###########################################
    if pool_size is None:
        pool_size = max(cpu_count() - 1, 1)

    args_list = [
        (cluster_id, unodes_idx_list, segments, unodes_coord)
        for cluster_id, unodes_idx_list in unodes_by_disc.items()
    ]

    with Pool(processes=pool_size) as pool:
        results = pool.map(build_graph_and_find_cycles, args_list)

    surface_loops = dict(results)
    return surface_loops


class HalfFace:
    """
    半面（Half-Face）数据结构。

    属性：
        cluster_id: int，所属结构面ID。
        pt_indices: Tuple[int]，该面的顶点索引列表。
        normal: Tuple[float]，该面的法向量。
    """

    def __init__(self, cluster_id, pt_indices, normal):
        self.cluster_id = cluster_id
        self.pt_indices = tuple(pt_indices)
        self.normal = tuple(normal)
        self.id = None

    def edges(self):
        """返回该面所有边的标准化形式（两端点索引排序）。"""
        return [tuple(sorted((self.pt_indices[i], self.pt_indices[(i + 1) % len(self.pt_indices)]))) for i in
                range(len(self.pt_indices))]

    def index(self, hf_idx):
        self.id = hf_idx

    def __hash__(self):
        return hash((self.cluster_id, self.pt_indices, self.normal))

    def __eq__(self, other):
        return (self.cluster_id, self.pt_indices, self.normal) == (other.cluster_id, other.pt_indices, other.normal)

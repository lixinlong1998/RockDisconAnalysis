import time
from scipy.spatial import cKDTree as KDTree  # 替换 sklearn 的 KDTree
from numba import njit, prange
from collections import defaultdict
import numpy as np
from src import Export


class Segment:
    def __init__(self, p1, p2, line_dir, surface_ids):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        if surface_ids:
            self.surface_ids = tuple(sorted(surface_ids))  # e.g., (3, 12)  这2个索引是cluster_id，指向discontinuity
        else:
            self.surface_ids = None
        self.dir = line_dir
        self.id = None  # 可附加唯一编号
        self.meta = {}  # 其他元数据如 parent segment, local index 等

    def as_array(self):
        return np.vstack([self.p1, self.p2])

    def length(self):
        return np.linalg.norm(self.p2 - self.p1)


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

    # export line_dirs, points
    start_points = points + 3 * line_dirs
    end_points = points - 3 * line_dirs
    points = []
    edges = []
    for i, pt_coord in enumerate(start_points):
        start = pt_coord
        end = end_points[i]

        # write data
        idx_start = len(points)
        idx_end = idx_start + 1
        points.append(start)
        points.append(end)
        edges.append([idx_start, idx_end])

    points = np.array(points, dtype=np.float64)
    edges = np.array(edges, dtype=np.int32)
    Export.write_ply_with_edges(
        r"D:\Research\20250313_RockFractureSeg\Experiments\Exp2_DiscontinuityCharacterization\test_line_dir.ply",
        points, edges)

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
                    if 1e-3 < radii[i] + radii[j] - dist_ij:
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

    # 验证
    discontinuity_pairs_distance = np.asarray(
        [(pair[0].ellip_a + pair[1].ellip_a) * extention - np.linalg.norm(pair[0].disc_center - pair[1].disc_center) for
         pair in discontinuity_pairs])

    return discontinuity_pairs


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
    # 此处使用的是结构面的中心，而不是一开始的平面方程，也就是说disc和拟合的平面方程可能不重合，除非从点云拟合平面，在平面中找到迹线后直接反投影得到其坐标，而不是通过索引返回那对点本身。
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
    coplanar_mask = np.isclose(norms, 0, atol=coplanar_tol)
    valid_mask = ~coplanar_mask
    line_dirs = np.zeros_like(raw_dirs)
    line_dirs[~coplanar_mask] = raw_dirs[~coplanar_mask] / norms[~coplanar_mask, None]  # 归一化后的方向
    assert raw_dirs.shape[1] == 3

    # 平面方程的 d = n·p
    d1s = np.einsum('ij,ij->i', n1s, p1s)
    d2s = np.einsum('ij,ij->i', n2s, p2s)

    # 初始化交点数组
    points_on_lines = np.full_like(n1s, np.nan)

    # 只处理有效平面对
    # 对每组有效的平面对单独处理
    for i in np.where(valid_mask)[0]:
        # 构造方程组 A x = b
        A = np.vstack([n1s[i], n2s[i]])  # (2, 3)
        b = np.array([
            np.dot(n1s[i], p1s[i]),  # n1·p1
            np.dot(n2s[i], p2s[i])  # n2·p2
        ])

        # 求解最小二乘问题
        point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        points_on_lines[i] = point

    return line_dirs, points_on_lines, coplanar_mask
    # #
    # for i in range(len(discontinuity_pairs)):
    #     if coplanar_mask[i]:
    #         continue
    #     A = np.column_stack([n1s[i], n2s[i], line_dirs[i]])  # (3, 3)
    #     b = np.array([d1s[i], d2s[i], 0.0])
    #     try:
    #         x = np.linalg.solve(A, b)
    #         points_on_lines[i] = x
    #     except np.linalg.LinAlgError:
    #         # 奇异矩阵时处理为 nan
    #         coplanar_mask[i] = True
    #         points_on_lines[i] = np.nan
    #         line_dirs[i] = 0.0
    # ## 判断此点线方程是否在该平面方程上
    # # 直线方向与平面faxiang
    # return line_dirs, points_on_lines, coplanar_mask


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
            Segment(coord_start, coord_end, line_dir_valid[i], (disc_pair_ids_valid[i, 0], disc_pair_ids_valid[i, 1])))
    return segments


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

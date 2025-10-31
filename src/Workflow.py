import struct
import time
import open3d as o3d
import pickle
import numpy as np
from src import Cluster
from src import Discontinuity
from src import PointCloud
import pandas as pd


#####################################################################################
#####################################################################################
#####################################################################################

def load_pointcloud_DSE(data_path):
    '''
    :param data_path: 文件路径，tab分隔、无表头
    :return: X, Y, Z, js, cl, A, B, C, D（为张量化字段），rock_pointcloud 对象
    '''
    starttime = time.perf_counter()
    # 读取数据（无表头，tab 分隔）
    data = np.loadtxt(data_path, delimiter='\t')

    # 提取所有 (js, cl) 并生成唯一 cluster_id
    js_cl_list = [(int(row[3]), int(row[4])) for row in data]
    unique_clusters = list(set(js_cl_list))
    cluster_id_dict = {key: idx for idx, key in enumerate(unique_clusters)}
    clusters_pointcloud = {key: PointCloud.RockPointCloud() for idx, key in enumerate(unique_clusters)}

    total_pointcloud = PointCloud.RockPointCloud()
    for point_id, row in enumerate(data):
        X, Y, Z = row[0:3]
        joint_id = int(row[3])
        joint_cluster_id = int(row[4])
        cluster_id = cluster_id_dict[(joint_id, joint_cluster_id)]
        R, G, B = 0, 0, 0
        a, b, c, d = row[5:9]

        total_pointcloud.add(X, Y, Z, point_id, joint_id, joint_cluster_id, cluster_id, R, G, B, a, b, c, d)

    # 提取感兴趣字段用于张量化处理
    print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — load point cloud to total_pointcloud.')
    return data, total_pointcloud, clusters_pointcloud, cluster_id_dict


def load_pointcloud_QFACET(data_path):
    """
    读取“数据格式2”（含表头、tab分隔）文件，并返回：
      data: (N,9) 兼容旧接口的 ndarray，列为 [X,Y,Z, js, cl, A,B,C,D]
      total_pointcloud: PointCloud.RockPointCloud
      clusters_pointcloud: dict[(js,cl)] -> RockPointCloud
      cluster_id_dict: dict[(js,cl)] -> combined_cluster_id
    规则（简化版）：
      - 正常簇(js!=-1, cl!=-1)：combined_cluster_id 按出现顺序连续编号（与原逻辑相同）
      - js==-1：cl 由 facet_id 去重后映射到 0,1,2,...；combined_cluster_id = facet_id
                 若 facet_id 无效(<0) 则该点仍记为噪声（combined_cluster_id=-1）
    """
    starttime = time.perf_counter()

    # 1) 读表 + 基础校验
    df = pd.read_csv(data_path, sep="\t")
    need = ["X", "Y", "Z", "cluster_id", "subcluster_id", "facet_id", "A", "B", "C", "D"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    # 2) 类型规范（无效一律置 -1 / NaN→-1）
    for col in ["X", "Y", "Z", "A", "B", "C", "D"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[["cluster_id", "subcluster_id", "facet_id"]] = df[["cluster_id", "subcluster_id", "facet_id"]].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)
    )

    # 3) 为 js==-1 的点，按 facet_id 生成新的 cl（0..M-1）
    neg_mask = (df["cluster_id"] == -1)
    valid_facet = df.loc[neg_mask & (df["facet_id"] >= 0), "facet_id"]
    uniq_facet = pd.unique(valid_facet)
    facet2cl = {int(fid): i for i, fid in enumerate(uniq_facet)}  # facet_id -> new_cl

    # 仅覆盖 js==-1 的 cl；其他保持原 cl 值
    df.loc[neg_mask, "subcluster_id"] = df.loc[neg_mask, "facet_id"].map(facet2cl).fillna(-1).astype(int)

    # 4) 生成 cluster_id_dict
    #    - 正常簇：连续编号（保持和你当前实现一致）:contentReference[oaicite:2]{index=2}
    #    - js==-1：(-1, new_cl) -> facet_id
    pairs_pos = list(
        df.loc[(df["cluster_id"] != -1) & (df["subcluster_id"] != -1), ["cluster_id", "subcluster_id"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    cluster_id_dict = {key: idx for idx, key in enumerate(pairs_pos)}  # 正常簇

    # 为 js==-1 的每个映射，指定 combined_cluster_id = facet_id（不占用连续编号）
    for fid, new_cl in facet2cl.items():
        cluster_id_dict[(-1, new_cl)] = int(fid)

    # 5) 构建容器
    clusters_pointcloud = {key: PointCloud.RockPointCloud() for key in cluster_id_dict.keys()}
    total_pointcloud = PointCloud.RockPointCloud()

    # 6) 写入点（含 combined_cluster_id 的新规则）
    X = df["X"].to_numpy(float);
    Y = df["Y"].to_numpy(float);
    Z = df["Z"].to_numpy(float)
    js = df["cluster_id"].to_numpy(int);
    cl = df["subcluster_id"].to_numpy(int)
    a = df["A"].to_numpy(float);
    b = df["B"].to_numpy(float);
    c = df["C"].to_numpy(float);
    d = df["D"].to_numpy(float)
    fid = df["facet_id"].to_numpy(int)
    R = G = B = 0

    n = len(df)
    for i in range(n):
        j = int(js[i]);
        k = int(cl[i])
        if j == -1:
            # 关键：js==-1 → combined_cluster_id = facet_id（若 facet_id 无效则仍记噪声）
            ccid = int(fid[i]) if int(fid[i]) >= 0 else -1
        elif k == -1:
            ccid = -1
        else:
            ccid = cluster_id_dict[(j, k)]

        total_pointcloud.add(float(X[i]), float(Y[i]), float(Z[i]),
                             int(i), j, k, int(ccid), R, G, B,
                             float(a[i]), float(b[i]), float(c[i]), float(d[i]))

        if ccid != -1:
            clusters_pointcloud[(j, k)].add(float(X[i]), float(Y[i]), float(Z[i]),
                                            int(i), j, k, int(ccid), R, G, B,
                                            float(a[i]), float(b[i]), float(c[i]), float(d[i]))

    print(f"[time cost]{time_cost_hms(time.perf_counter() - starttime)} - load point cloud (format2).")

    # 7) 兼容旧 ndarray（列顺序与原版一致）:contentReference[oaicite:3]{index=3}
    data = np.column_stack([X, Y, Z, js, cl, a, b, c, d]).astype(np.float64)
    return data, total_pointcloud, clusters_pointcloud, cluster_id_dict


def get_clusters(data, total_pointcloud, clusters_pointcloud, cluster_id_dict):
    '''

    :param data:
    :param total_pointcloud:
    :param clusters_pointcloud: dict{(int, int):PointCloud.RockPointCloud()},键为 (js, cl),值为cluster对应的点云。
    :param cluster_id_dict:
    :return: clusters类
    '''
    testset = [(4, 39), (2, 7), (1, 37), (5, 81), (2, 87), (4, 48), (6, 80), (2, 196)]  # ]]]]]]]]]]]]]]]]]]]]]]]]]

    starttime = time.perf_counter()
    # 遍历data的每行点数据，将点分配到与cluster_id=(js, cl)对应的list容器中
    for point_idx, row in enumerate(data):
        js = int(row[3])  # 第 4 列
        cl = int(row[4])  # 第 5 列
        if cl == -1:
            continue
        clusters_pointcloud[(js, cl)].append(total_pointcloud.points[point_idx])

    # 从clusters_pointcloud中构建cluster类
    clusters = Cluster.Clusters()
    for key, cluster_id in cluster_id_dict.items():
        # # test part
        # if key not in testset:
        #     continue

        # get point cloud and plane_params
        rock_points = clusters_pointcloud[key]
        plane_params = rock_points.points[0].plane_paras

        # initial inlier_ratio
        inlier_ratio = 0

        # append cluster object
        clusters.add(key[0], key[1], cluster_id, rock_points, plane_params, inlier_ratio)

    print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — create clusters object of RockSeg3D.')
    return clusters


def ransac_planarity_filter(clusters):
    threshold_inliers_ratio = 0.4
    threshold_inliers_number = 50
    test_scale_with_factor = []
    for cluster in clusters.clusters:

        # get the coordinates of point cloud
        points_coord = np.asarray([pt.coord for pt in cluster.rock_points.points])

        # get the scale of point cloud
        starttime1 = time.perf_counter()
        scale_with_factor, scale = estimate_adaptive_threshold(points_coord, scale_factor=0.05)
        # print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime1)} — estimate_adaptive_threshold.')

        # test
        test_scale_with_factor.append(scale_with_factor)
        print(scale_with_factor)

        # 基于点云尺度，自适应调整RANSAC平面拟合的容差阈值distance_threshold
        starttime2 = time.perf_counter()
        inliers, outliers, inlier_mask, plane_params = fit_plane_ransac(points_coord,
                                                                        distance_threshold=scale_with_factor)
        # print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime2)} — fit_plane_ransac.')

        # 检查内点率阈值和内点数量阈值
        inlier_ratio = len(inliers) / (len(inliers) + len(outliers))
        if inlier_ratio <= threshold_inliers_ratio or len(inliers) <= threshold_inliers_number:
            cluster.valid = False
            print(
                f"False!  内点数: {len(inliers)}, 外点数: {len(outliers)}, 内点率: {inlier_ratio},{cluster.joint_id}-{cluster.joint_cluster_id} ")
            continue  # 跳过非平面或小平面
        else:
            print(
                f"内点数: {len(inliers)}, 外点数: {len(outliers)}, 内点率: {inlier_ratio},{cluster.joint_id}-{cluster.joint_cluster_id} ")

        # # calculate planarity score of cluster points
        # planarity_score, uniform_score = check_cluster_planarity(points_coord, k=30)
        # scores = compute_planarity_score(points_coord, k=30)
        # # 分离内外点（20% 最不平面为外点）
        # inliers, outliers = separate_inliers_outliers(points_coord, scores, method="quantile", threshold=0.2)
        # print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime0)} — remove outlier.')

        # 可视化
        # pcd_in = o3d.geometry.PointCloud()
        # pcd_in.points = o3d.utility.Vector3dVector(inliers)
        # pcd_in.paint_uniform_color([0, 1, 0])  # 绿色
        # pcd_out = o3d.geometry.PointCloud()
        # pcd_out.points = o3d.utility.Vector3dVector(outliers)
        # pcd_out.paint_uniform_color([1, 0, 0])  # 红色
        # o3d.visualization.draw_geometries([pcd_in, pcd_out])

        # update cluster attributes
        # print(inlier_mask)
        # print(np.all(~inlier_mask))
        cluster.inlier_mask = inlier_mask
        cluster.inlier_number = len(inliers)
        cluster.inlier_ratio = inlier_ratio
        cluster.valid = True
        cluster.centroid = cluster.get_centroid()
        cluster.plane_params = np.asarray(plane_params, dtype=np.float64)
        cluster.normal = plane_params[:3] / np.linalg.norm(plane_params[:3])

    # test

    return clusters


def estimate_adaptive_threshold(points, scale_factor=0.02):
    """
    基于PCA主轴尺度动态设置拟合容差阈值（例如用于RANSAC）。

    参数:
        points (np.ndarray): shape (N, 3) 的三维点云
        scale_factor (float): 与点云尺度成比例的系数（默认2%）

    返回:
        distance_threshold (float): 平面拟合距离容差
    """
    # assert points.ndim == 2 and points.shape[1] == 3, "输入必须是(N, 3)三维数组"
    #
    # # 去中心化
    # centered = points - np.mean(points, axis=0)
    #
    # # PCA 主轴尺度（协方差矩阵特征值）
    # cov = np.cov(centered.T)
    # eigvals = np.linalg.eigvalsh(cov)  # 已排序（升序）
    #
    # # 使用最大空间尺度作为参考
    # first_max_extent = np.sqrt(eigvals[-1])
    # second_max_extent = np.sqrt(eigvals[-2])
    #
    # scale_area = first_max_extent * second_max_extent  # 矩形面积

    # 基于aabb外包盒的对角线长度来评估点云集的尺寸
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    diag_len = np.linalg.norm(max_corner - min_corner)

    tolerance = a * log(1 + b * diag_len)  # a,b为经验参数

    return tolerance, diag_len

    # # 阈值与尺度成正比
    # distance_threshold = scale_area * scale_factor
    # return distance_threshold, scale_area


def fit_plane_ransac(point_array, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    '''
    使用 RANSAC 拟合空间平面并返回平面参数、内点和外点   inlier_mask😢

    参数:
    - point_array: np.ndarray, shape (N, 3)，输入的点云坐标
    - distance_threshold: float，内点的最大距离阈值
    - ransac_n: int，RANSAC 每次拟合所需的最小样本数（拟合平面为3）
    - num_iterations: int，RANSAC 最大迭代次数

    返回:
    - plane_model: list[float], [a, b, c, d] 平面方程 ax + by + cz + d = 0
    - inliers: np.ndarray, shape (M, 3)，拟合平面上的内点
    - outliers: np.ndarray, shape (N-M, 3)，不在平面上的外点
    '''

    # 转为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)

    # RANSAC 拟合平面
    plane_params, inlier_idxs = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )  # inlier_idxs中的每个数据是point_array的索引，指向了内点

    # 将索引转换为mask
    inlier_mask = np.zeros(len(point_array), dtype=bool)
    inlier_mask[inlier_idxs] = True

    # 提取内点和外点
    inliers = point_array[inlier_mask]
    outliers = point_array[~inlier_mask]

    return inliers, outliers, inlier_mask, plane_params


def check_cluster_planarity(points, k=30):
    '''
    对整个点集计算整体平面性评分
    输入：
        points: (N, 3) np.ndarray
    输出：
        planarity_score: float，越小越接近平面
        normal: 法向量 (3,)
    '''
    pts_centered = points - points.mean(axis=0)
    cov = pts_centered.T @ pts_centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]  # λ1 ≥ λ2 ≥ λ3

    planarity_score = eigvals[2] / eigvals.sum()  # 越小越平面
    # normal = eigvecs[:, 0]  # 最小特征值对应法向

    """
    计算每个点的局部点密度（k邻域反距离均值）。
    输入：
        points: (N,3) ndarray，点集
        k: 邻域点数量
    输出：
        densities: (N,) ndarray，密度指标（数值越大密度越低）
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    # 忽略自身点
    avg_distance = np.mean(distances[:, 1:], axis=1)
    density = 1 / (avg_distance + 1e-8)

    # score: float，密度均匀性得分（值越小表示越均匀）
    uniform_score = np.std(density) / np.mean(density)

    return planarity_score, uniform_score


def compute_planarity_score(points, k=30):
    '''
    计算每个点的平面性得分（λ₂ / λ₃）
    输入:
        points: (N,3) 点云坐标
        k: 邻域大小
    输出:
        planarity_scores: (N,) array，越大越平面
    '''
    from sklearn.neighbors import NearestNeighbors
    scores = np.zeros(len(points))
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    indices = nbrs.kneighbors(return_distance=False)

    for i, idx in enumerate(indices):
        neighbors = points[idx] - points[idx].mean(axis=0)
        cov = neighbors.T @ neighbors
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # λ1 ≥ λ2 ≥ λ3
        if eigvals[1] != 0:
            scores[i] = (eigvals[1] - eigvals[2]) / eigvals[0]  # 可替换为 λ₂ / λ₃ 或其他指标
        else:
            scores[i] = 0

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    avg_distance = np.mean(distances[:, 1:], axis=1)
    densitys = 1 / (avg_distance + 1e-8)

    return scores + densitys


def separate_inliers_outliers(points, planarity_scores, method="quantile", threshold=0.2):
    '''
    根据平面性得分分离内点与外点

    输入：
        points: (N,3) 点云坐标
        planarity_scores: (N,) 平面性得分
        method: "quantile"（分位数）或 "std"（标准差）
        threshold: 若 method="quantile"，则表示分位数（如 0.2）；若 method="std"，表示标准差倍数（如 1.0）

    输出：
        inlier_points: 平面性较高的点
        outlier_points: 非平面区域或异常点
    '''
    if method == "quantile":
        cutoff = np.quantile(planarity_scores, threshold)
        mask = planarity_scores >= cutoff
    elif method == "std":
        mean = np.mean(planarity_scores)
        std = np.std(planarity_scores)
        mask = planarity_scores >= (mean - threshold * std)
    else:
        raise ValueError("Unsupported method")

    return points[mask], points[~mask]


def get_discontinuitys(clusters):
    def npget_dip_dir(np_clusters_normal):
        '''
        'npget' means using numpy tensor to facilitate the process.

        :param np_clusters_planeparas: numpy array with shape: (N, 3)，归一化后的A_nol, B_nol, C_nol
        :return: dip_direction, direction: numpy array with shape: (N, 2)，倾角、倾向
        '''
        a = np_clusters_normal[:, 0]
        b = np_clusters_normal[:, 1]
        c = np_clusters_normal[:, 2]
        # D：是平面与原点位置关系有关的偏移量，不影响方向，因此不参与单位化

        # 倾角 dip = arccos(|c|)，单位为度
        dips = np.degrees(np.arccos(np.abs(c)))  # 结果范围：[0, 90]

        # 倾向 strike = atan2(-b, a)，结果范围 [0, 360)
        strikes = np.degrees(np.arctan2(-b, a))
        strikes = np.where(strikes < 0, strikes + 360, strikes)

        # 输出 shape 校验（可选）
        print('A_nor.shape:', a.shape)
        print('dips.shape:', dips.shape)

        # 合并输出
        dip_dir = np.stack([dips, strikes], axis=1)  # (N, 2)
        return dip_dir  # 每一行是 [dip, strike]

    starttime = time.perf_counter()

    # calculate dip dir with numpy tensor
    np_clusters_normal = np.asarray([cluster.normal for cluster in clusters.clusters])
    np_dip_dir = npget_dip_dir(np_clusters_normal)

    discontinuitys = Discontinuity.Discontinuitys()
    for idx, cluster in enumerate(clusters.clusters):
        disc = Discontinuity.Discontinuity.from_cluster(cluster)
        disc.dip = np_dip_dir[idx, 0]
        disc.strike = np_dip_dir[idx, 1]
        disc.type = 'free_surface'
        discontinuitys.add(disc)

    print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — create discontinuitys from clusters.')
    return discontinuitys


def time_cost_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} h {m} min {s:.2f} sec"


###################################################
###################################################
###################################################
def get_cluster_id(cluster_pointcloud):
    cluster_id_by_key = {}
    cluster_key_by_id = {}
    for id, (key, point) in enumerate(cluster_pointcloud.items()):
        cluster_id_by_key[key] = id
        cluster_key_by_id[id] = key

    # with open(save_path_1, "wb") as f:
    #     pickle.dump(cluster_key_to_ids, f)
    #
    # with open(save_path_2, "wb") as f:
    #     pickle.dump(cluster_ids_to_key, f)

    return cluster_id_by_key, cluster_key_by_id


def get_cluster_planeparas(data, cluster_point, save_path):
    cluster_planeparas = {}
    for key, point in cluster_point.items():
        cluster_planeparas[key] = data[point[0], 5:9]

    with open(save_path, "wb") as f:
        pickle.dump(cluster_planeparas, f)

    return cluster_planeparas


def get_plane_params(data, cluster_point, js_val, cl_val):
    # 找到所有匹配的索引
    indices = cluster_point.get((js_val, cl_val), [])
    if indices:
        i = indices[0]  # 取第一个匹配索引
        return data[i, 5:9]  # 第6-9列为 a, b, c, d
    else:
        return None


def get_strike_dip(A, B, C):
    # 单位化法向量
    norm = np.sqrt(A ** 2 + B ** 2 + C ** 2)
    a, b, c = A / norm, B / norm, C / norm

    # 倾角（Dip）
    dip = np.degrees(np.arccos(abs(c)))  # 注意取abs保证正值

    # 倾向（Strike），注意 atan2(-B, A)
    strike = np.degrees(np.arctan2(-b, a))
    if strike < 0:
        strike += 360

    return strike, dip


def get_trace_plane(cluster_point: dict, dip_dict: dict, save_path):
    """
    给定三维点及其法向平面参数，将点投影至平面后，使用“两次最远点搜索”法找到trace的两个端点。

    :param cluster_point : dict，键为(js, cl)，值为点索引列表
    :param dip_dict       : dict，包含法向量信息（A,B,C）
    :param save_path      : 路径，保存trace_dict结果
    :return: trace_dict : dict，键为(js, cl)，值为字典，含长度和端点坐标
            trace_dict[(js, cl)] = {
            'length': max_len,
            'start': coord_start.tolist(),
            'end': coord_end.tolist(),
            'start_idx': int(idx_start),
            'end_idx': int(idx_end)
        }
    """

    def project_points_to_plane(points, normal):
        """
        将三维点投影到给定法向量定义的平面上
        返回投影后的二维坐标和平面局部坐标系基向量（u, v）
        """
        normal = normal / np.linalg.norm(normal)
        # 找一个不平行于法向的向量构造局部平面坐标系
        arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(normal, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        projected_2d = np.dot(points, np.vstack([u, v]).T)
        return projected_2d, u, v

    def double_farest_search(projected_2d):
        center = projected_2d.mean(axis=0)
        dists_to_center = np.linalg.norm(projected_2d - center, axis=1)
        idx1 = np.argmax(dists_to_center)  # 边缘点1在 projected_2d 中的索引

        dists_to_idx1 = np.linalg.norm(projected_2d - projected_2d[idx1], axis=1)
        idx2 = np.argmax(dists_to_idx1)  # 边缘点2

        # 计算距离
        max_len = np.max(dists_to_idx1)

        idx_start = point_indices[idx1]
        idx_end = point_indices[idx2]
        return max_len, idx_start, idx_end

    trace_dict = {}
    for key, point_indices in cluster_point.items():
        if key not in dip_dict:
            continue
        A_nol, B_nol, C_nol = dip_dict[key][2:5]
        normal = np.array([A_nol, B_nol, C_nol])
        points = data[point_indices, 0:3]  # 未知原因导致基于point_indices提取的points中都是同一个点，可能是多个点都是同一个坐标

        # print(points)
        # 将点投影到对应平面
        projected_2d, u, v = project_points_to_plane(points, normal)

        if len(projected_2d) < 2:
            print('无法计算距离')
            continue  # 无法计算距离

        # 使用“两次最远点搜索”法找到trace的两个端点
        max_len, idx_start, idx_end = double_farest_search(projected_2d)
        if max_len == 0:
            # print(f'max_len:{max_len}       {point_indices}')
            continue
            # print(point_indices)
            # print(points)
            # print(u, v)
            # print(projected_2d)
        if max_len > 1:
            print(f'max_len:{max_len}')

        coord_start = data[idx_start, 0:3]
        coord_end = data[idx_end, 0:3]

        trace_dict[key] = {
            'length': max_len,
            'start': coord_start.tolist(),
            'end': coord_end.tolist(),
            'start_idx': int(idx_start),
            'end_idx': int(idx_end)
        }

    with open(save_path, "wb") as f:
        pickle.dump(trace_dict, f)

    return trace_dict


# ================================export
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


def export_trace_to_ply(trace_dict, save_path):
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

    for trace in trace_dict.values():
        start = trace['start']
        end = trace['end']
        idx_start = len(points)
        idx_end = idx_start + 1
        points.append(start)
        points.append(end)
        edges.append([idx_start, idx_end])

    points = np.array(points, dtype=np.float64)
    edges = np.array(edges, dtype=np.int32)

    write_ply_with_edges(save_path, points, edges)

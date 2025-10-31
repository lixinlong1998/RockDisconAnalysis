import struct
import laspy
from itertools import combinations
import os
import pickle
from collections import defaultdict
import alphashape
import numpy as np
from shapely.geometry import Point, Polygon
import pyvista as pv
from scipy.spatial import ConvexHull, Delaunay


class Point3D:
    def __init__(self, X: float, Y: float, Z: float, point_id: int, joint_id: int, joint_cluster_id: int,
                 cluster_id: int, R, G, B, a, b, c, d):
        self.coord = np.asarray([X, Y, Z], dtype=np.float64)
        self.point_id = point_id
        self.joint_id = joint_id
        self.joint_cluster_id = joint_cluster_id
        self.cluster_key = (self.joint_id, self.joint_cluster_id)
        self.cluster_id = cluster_id
        self.color = np.asarray([R, G, B], dtype=np.uint8)
        self.plane_paras = np.asarray([a, b, c, d], dtype=np.float64)


class RockPointCloud:
    def __init__(self):
        self.points = []
        self.number = len(self.points)
        self.where = 'save_path'

    def add_point(self, X: float, Y: float, Z: float, point_id: int, joint_id: int, joint_cluster_id: int,
                  cluster_id: int, R, G, B, a, b, c, d):
        self.points.append(Point3D(X, Y, Z, point_id, joint_id, joint_cluster_id, cluster_id, R, G, B, a, b, c, d))

    def delete_point(self, point_id_list):
        self.points = [p for p in self.points if p.point_id not in point_id_list]

    def rewrite_point(self, point_id, **kwargs):
        for i, p in enumerate(self.points):
            if p.point_id == point_id:
                self.points[i] = Point3D(
                    kwargs.get('X', p.coord[0]), kwargs.get('Y', p.coord[1]), kwargs.get('Z', p.coord[2]),
                    point_id,
                    kwargs.get('joint_id', p.joint_id),
                    kwargs.get('joint_cluster_id', p.joint_cluster_id),
                    kwargs.get('cluster_id', p.cluster_id),
                    kwargs.get('R', p.color[0]), kwargs.get('G', p.color[1]), kwargs.get('B', p.color[2]),
                    kwargs.get('a', p.plane_paras[0]), kwargs.get('b', p.plane_paras[1]),
                    kwargs.get('c', p.plane_paras[2]), kwargs.get('d', p.plane_paras[3])
                )
                break

    def update_index(self):
        for i, p in enumerate(self.points):
            p.point_id = i

    def load(self, path, format='pkl'):
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return

        if format == 'pkl':
            with open(path, 'rb') as f:
                self.points = pickle.load(f)
        elif format == 'npz':
            data = np.load(path)
            self.points = []
            for i in range(data['coords'].shape[0]):
                X, Y, Z = data['coords'][i]
                R, G, B = data['colors'][i]
                point_id, joint_id, joint_cluster_id, cluster_id = data['ids'][i]
                a, b, c, d = data['planes'][i]
                self.add_point(X, Y, Z, point_id, joint_id, joint_cluster_id, cluster_id, R, G, B, a, b, c, d)
        elif format == 'ply':
            from plyfile import PlyData
            plydata = PlyData.read(path)
            vertex = plydata['vertex']
            self.points = []
            for i in range(len(vertex)):
                self.add_point(
                    vertex[i]['x'], vertex[i]['y'], vertex[i]['z'],
                    vertex[i]['point_id'], vertex[i]['joint_id'],
                    vertex[i]['joint_cluster_id'], vertex[i]['cluster_id'],
                    vertex[i]['red'], vertex[i]['green'], vertex[i]['blue'],
                    vertex[i]['a'], vertex[i]['b'], vertex[i]['c'], vertex[i]['d']
                )
        else:
            print(f"Unsupported load format: {format}")

    def export(self, path, format='ply'):
        '''
        :param path: save path
        :param format: ply_bin, ply_txt, pkl, npz, las
        :return: saved points with given format
        '''
        if len(self.points) == 0:
            print("No points to export.")
            return

        coords = np.array([p.coord for p in self.points])
        colors = np.array([p.color for p in self.points])
        ids = np.array([[p.point_id, p.joint_id, p.joint_cluster_id, p.cluster_id] for p in self.points])
        planes = np.array([p.plane_paras for p in self.points])

        if format == 'ply_bin' or format == 'ply_txt':
            from plyfile import PlyData, PlyElement

            vertex_data = np.array(
                [tuple(coords[i]) + tuple(colors[i]) + tuple(ids[i]) + tuple(planes[i])
                 for i in range(len(coords))],
                dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('point_id', 'i4'), ('joint_id', 'i4'),
                    ('joint_cluster_id', 'i4'), ('cluster_id', 'i4'),
                    ('a', 'f4'), ('b', 'f4'), ('c', 'f4'), ('d', 'f4'),
                ]
            )

            el = PlyElement.describe(vertex_data, 'vertex')
            PlyData([el], text=(format == 'ply_txt')).write(path)
            print(f"Exported to {path} as {'ASCII' if format == 'ply_txt' else 'binary'} PLY.")

        elif format == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(self.points, f)
            print(f"Exported to {path} as Pickle (.pkl) file.")

        elif format == 'npz':
            np.savez(path, coords=coords, colors=colors, ids=ids, planes=planes)
            print(f"Exported to {path} as compressed NumPy (.npz) file.")

        elif format == 'las':
            if laspy is None:
                print("Error: laspy module not available. Cannot export LAS.")
                return

            header = laspy.LasHeader(point_format=3, version="1.2")
            las = laspy.LasData(header)

            las.x = coords[:, 0]
            las.y = coords[:, 1]
            las.z = coords[:, 2]
            las.red = colors[:, 0].astype(np.uint16)
            las.green = colors[:, 1].astype(np.uint16)
            las.blue = colors[:, 2].astype(np.uint16)

            las.write(path)
            print(f"Exported to {path} as LAS (.las) file.")
        else:
            print(f"Unsupported export format: {format}")





def load_data(data_path):
    '''
    :param data_path:
    :return: X, Y, Z, js, cl, D, B, C, D
    '''
    # 读取数据（无表头，tab 分隔）
    data = np.loadtxt(data_path, delimiter='\t')
    return data


def get_cluster_point(data, save_path):
    """
    直接遍历 data 的每一行，构建 cluster_point。

    Parameters
    ----------
    data : np.ndarray
        N×M 数组，至少包含第 3 列 js、第 4 列 cl。
    save_path : str
        pickle 文件保存路径。

    Returns
    -------
    cluster_point : dict[(int, int), list[int]]
        键为 (js, cl)，值为该组对应的行索引列表。
    """
    cluster_point = defaultdict(list)
    # 直接对 data 每行循环
    for point_idx, row in enumerate(data):
        js = int(row[3])  # 第 4 列
        cl = int(row[4])  # 第 5 列
        cluster_point[(js, cl)].append(point_idx)

    # print(cluster_point[(6, 2)])  # for test

    with open(save_path, "wb") as f:
        pickle.dump(cluster_point, f)

    return cluster_point


def get_cluster_id(cluster_point):
    cluster_id_by_key = {}
    cluster_key_by_id = {}
    for id, (key, point) in enumerate(cluster_point.items()):
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


def get_dip_dict(cluster_planeparas, save_path):
    '''
    :param cluster_planeparas: dict{(js, cl): [A, B, C, D], ...}
    :param save_path:
    :return:dip_dict, dict, (js,cl), [dips[i], strikes[i], a[i], b[i], c[i]]
    '''
    keys = list(cluster_planeparas.keys())
    values = np.array(list(cluster_planeparas.values()))  # shape: (N, 4)
    A_col = values[:, 0]
    B_col = values[:, 1]
    C_col = values[:, 2]
    D_col = values[:, 3]
    print('A_col.shape:', A_col.shape)
    # 向量化计算法向量单位化
    normals = np.stack([A_col, B_col, C_col], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_unit = normals / norms  # (N, 3)

    a, b, c = normals_unit[:, 0], normals_unit[:, 1], normals_unit[:, 2]
    print('a.shape:', a.shape)

    # 倾角 dip = arccos(|c|)  → degrees
    dips = np.degrees(np.arccos(np.abs(c)))

    # 倾向 strike = atan2(-b, a)，结果范围 [0, 360)
    strikes = np.degrees(np.arctan2(-b, a))
    strikes = np.where(strikes < 0, strikes + 360, strikes)

    # 构建 dip_dict：每个 (js, c) 对应其 dip 和 strike（取 group 中第一个索引值即可）
    dip_dict = {}
    for i, key in enumerate(keys):
        dip_dict[key] = [dips[i], strikes[i], a[i], b[i], c[i]]

    with open(save_path, "wb") as f:
        pickle.dump(dip_dict, f)

    return dip_dict


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


if __name__ == '__main__':
    '''
    values:
    data, nparrary

    cluster_vertex, (js,cl), [idx,...]
    
    cluster_id_by_key, (js,cl), id
    
    cluster_key_by_id, id, (js,cl)
    
    cluster_planeparas, (js,cl), [A,B,C,D]
    
    dip_dict, (js,cl), [dips[i], strikes[i], a[i], b[i], c[i]]
    
    trace_dict, (js, cl), {
                            'length': max_len,
                            'start': coord_start.tolist(),
                            'end': coord_end.tolist(),
                            'start_idx': int(idx_start),
                            'end_idx': int(idx_end)
                        }
                        
    '''
    # input
    data_path = r'D:\Research\20250313_RockFractureSeg\Experiments\Exp2_DiscontinuityCharacterization\G3033_9_Part2_ClusterAnalysis xyz-js-c-abcd.txt'
    # output
    cluster_point_path = os.path.join(os.path.dirname(data_path), 'G3033_9_Part2_ClusterAnalysis_cluster_point.pkl')
    cluster_planeparas_path = os.path.join(os.path.dirname(data_path),
                                           'G3033_9_Part2_ClusterAnalysis_cluster_planeparas.pkl')
    dip_dict_path = os.path.join(os.path.dirname(data_path), 'G3033_9_Part2_ClusterAnalysis_dip_dict.pkl')
    trace_dict_path = os.path.join(os.path.dirname(data_path), 'G3033_9_Part2_ClusterAnalysis_trace_dict.pkl')
    trace_dict_path_ply = os.path.join(os.path.dirname(data_path), 'G3033_9_Part2_ClusterAnalysis_TraceSegments.ply')
    disk_dict_path = os.path.join(os.path.dirname(data_path), 'G3033_9_Part2_ClusterAnalysis_DiscontinuityDisks.pkl')
    disk_dict_path_ply = os.path.join(os.path.dirname(data_path),
                                      'G3033_9_Part2_ClusterAnalysis_DiscontinuityDisks.ply')

    # workflow
    data = load_data(data_path)  # 以point为行
    cluster_point = get_cluster_point(data, cluster_point_path)  # 以cluster为行
    cluster_id_by_key, cluster_key_by_id = get_cluster_id(cluster_point)  # cluster的键索引和序号索引相互转换
    cluster_planeparas = get_cluster_planeparas(data, cluster_point, cluster_planeparas_path)  # cluster对应的平面参数
    dip_dict = get_dip_dict(cluster_planeparas, dip_dict_path)  # cluster对应的倾向倾角以及自归一化的平面法向向量(a,b,c)
    trace_dict = get_trace_plane(cluster_point, dip_dict, trace_dict_path)  # 根据cluster散点提取迹线线段
    export_trace_to_ply(trace_dict, trace_dict_path_ply)  # 导出迹线线段为vertex+edge的ply数据
    # disk_dict = build_disks_from_traces(cluster_planeparas, trace_dict, disk_dict_path)  # 基于迹线生成结构面圆盘网络
    # export_disk_to_ply(disk_dict, disk_dict_path_ply)  # 导出结构面圆盘网络的ply数据
    print(get_plane_params(data, cluster_point, 1, 52))

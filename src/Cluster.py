import os
import pickle
import numpy as np
from src import PointCloud


class Cluster:
    def __init__(self, joint_id, joint_cluster_id, cluster_id, rock_points, plane_params, inlier_ratio):
        self.joint_id = joint_id
        self.joint_cluster_id = joint_cluster_id
        self.cluster_id = cluster_id
        self.rock_points = rock_points  # 自定义点云类
        self.inlier_mask = np.zeros(rock_points.number, dtype=bool)  # rock_points序列镜像的布尔数组
        self.inlier_number = rock_points.number
        self.inlier_ratio = inlier_ratio
        self.valid = False
        self.centroid = self.get_centroid()
        self.plane_params = np.asarray(plane_params, dtype=np.float64)
        self.normal = self.plane_params[:3] / np.linalg.norm(self.plane_params[:3])

    def __repr__(self):
        return f"Cluster(js={self.joint_id}, cl={self.joint_cluster_id}, points={len(self.rock_points.points)})"

    def get_centroid(self):
        np_points = np.asarray([point.coord for point in self.rock_points.points])

        if np.any(self.inlier_mask):
            np_points = np_points[self.inlier_mask]

        # 由于未知原因，有些cluster由几个相同的点构成，所以需要检查。
        if np.all(np.all(np_points == np_points[0], axis=1)):
            self.valid = False
            return None
        else:
            self.valid = True
            return np.mean(np_points, axis=0)

    @property
    def projections_uv(self):
        '''
        构造与法向量 normal 垂直的两个单位向量 u, v，形成局部二维投影平面基底
        :return: (u, v) as 3D unit vectors, shape (3,), (3,)
        '''
        n = self.normal
        # 任取一个不与 n 平行的向量进行叉乘
        if np.allclose(n, [0, 0, 1], atol=1e-6):
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 0.0, 1.0])
        u = np.cross(n, a)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return u, v

    @property
    def projections(self):
        '''
        将三维点投影到以centroid为原点、normal为法向量的二维平面上
        :return: N x 2 array，表示每个点在 (u, v) 平面中的二维坐标
        '''
        u, v = self.projections_uv
        np_points = np.asarray([point.coord for point in self.rock_points.points])  # shape (N, 3)
        shifted = np_points - self.centroid  # 平移到以centroid为原点
        x_proj = np.dot(shifted, u)  # 投影到 u
        y_proj = np.dot(shifted, v)  # 投影到 v
        return np.stack([x_proj, y_proj], axis=1)  # shape (N, 2)

    def project_2d_to_3d(self, points_2d):
        """
        将二维平面上的点映射回三维空间点。
        参数:
            points_2d: np.ndarray
                单个二维点，shape=(2,)；或多个二维点，shape=(N, 2)
        返回:
            np.ndarray
                对应的三维点，shape=(3,) 或 (N, 3)
        """
        is_single_point = (points_2d.ndim == 1)

        # 保证至少为二维，(2,) -> (1, 2)
        points_2d = np.atleast_2d(points_2d)

        u, v = self.projections_uv  # shape (3,)
        centroid = self.centroid  # shape (3,)

        x = points_2d[:, 0]
        y = points_2d[:, 1]

        points_3d = centroid[None, :] + x[:, None] * u[None, :] + y[:, None] * v[None, :]

        # 如果原始输入是一维点，则返回一维三维点
        return points_3d[0] if is_single_point else points_3d

    def project_3d_to_2d(self, points_3d):
        """
        将三维空间中的点投影到局部二维平面。
        参数:
            points_3d: np.ndarray
                单个三维点，shape=(3,)；或多个三维点，shape=(N, 3)
        返回:
            np.ndarray
                对应的二维点，shape=(2,) 或 (N, 2)
        """
        is_single_point = (points_3d.ndim == 1)

        points_3d = np.atleast_2d(points_3d)

        u, v = self.projections_uv
        origin = self.centroid

        vecs = points_3d - origin[None, :]
        x = np.einsum('ij,j->i', vecs, u)
        y = np.einsum('ij,j->i', vecs, v)

        points_2d = np.column_stack((x, y))

        return points_2d[0] if is_single_point else points_2d


class Clusters:
    def __init__(self):
        self.clusters = []
        self.where = 'save_path'

    @property
    def number(self):
        return len(self.clusters)

    def append(self, Cluster):
        self.clusters.append(Cluster)

    def add(self, joint_id, joint_cluster_id, cluster_id, rock_points, plane_params, inlier_ratio):
        self.clusters.append(Cluster(joint_id, joint_cluster_id, cluster_id, rock_points, plane_params, inlier_ratio))

    def delete(self, cluster_id_list):
        self.clusters = [clu for clu in self.clusters if clu.cluster_id not in cluster_id_list]

    def rewrite_cluster(self, cluster_id, **kwargs):
        for clu in self.clusters:
            if clu.cluster_id == cluster_id:
                if 'points' in kwargs:
                    clu.points = PointCloud.RockclusterCloud(kwargs['points'])
                    clu.centroid = np.mean(kwargs['points'], axis=0)
                if 'plane_params' in kwargs:
                    clu.plane_params = np.asarray(kwargs['plane_params'], dtype=np.float64)
                    clu.normal = clu.plane_params[:3] / np.linalg.norm(clu.plane_params[:3])
                break
        else:
            print(f"No cluster found with id: {cluster_id}")

    def update_index(self):
        for i, clu in enumerate(self.clusters):
            clu.cluster_id = i

    def load(self, path, format='pkl'):
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return

        if format == 'pkl':
            with open(path, 'rb') as f:
                self.clusters = pickle.load(f)

        elif format == 'npz':
            data = np.load(path, allow_pickle=True)
            cluster_data = data['clusters']
            self.clusters = []
            for clu_dict in cluster_data:
                clu = Cluster(
                    clu_dict['joint_id'],
                    clu_dict['joint_cluster_id'],
                    clu_dict['cluster_id'],
                    clu_dict['points'],
                    clu_dict['plane_params']
                )
                self.clusters.append(clu)

        else:
            print(f"Unsupported load format: {format}")

    def export(self, path, format='pkl'):
        '''
        :param path: save path
        :param format: pkl, npz
        '''
        if len(self.clusters) == 0:
            print("No clusters to export.")
            return

        if format == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(self.clusters, f)
            print(f"Exported to {path} as Pickle (.pkl) file.")

        elif format == 'npz':
            cluster_data = []
            for clu in self.clusters:
                cluster_data.append({
                    'joint_id': clu.joint_id,
                    'joint_cluster_id': clu.joint_cluster_id,
                    'cluster_id': clu.cluster_id,
                    'points': np.asarray(clu.points),  # Assuming RockclusterCloud behaves like a NumPy array
                    'plane_params': clu.plane_params
                })
            np.savez_compressed(path, clusters=cluster_data)
            print(f"Exported to {path} as NumPy (.npz) file.")

        else:
            print(f"Unsupported export format: {format}")

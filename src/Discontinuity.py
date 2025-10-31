import pickle
import pandas as pd
import csv
import numpy as np
from scipy.spatial import ConvexHull
import numpy as np
from scipy.optimize import leastsq
from sklearn.decomposition import PCA
import pyvista as pv
from shapely.geometry import MultiPoint
from shapely.ops import triangulate
from scipy.spatial import Delaunay
from src import Cluster
from shapely.geometry import Polygon, MultiLineString
import alphashape
from shapely.ops import triangulate
import numpy as np


class Discontinuity(Cluster.Cluster):
    def __init__(self, joint_id, joint_cluster_id, cluster_id, rock_points, plane_params, inlier_ratio):
        super().__init__(joint_id, joint_cluster_id, cluster_id, rock_points, plane_params, inlier_ratio)
        self.valid = self.valid  # 父类Cluster构造中已经计算
        self.centroid = self.centroid  # 父类Cluster已有
        self.normal = self.normal  # 父类Cluster已有

        # Discontinuity自己扩展的属性
        self.dip = None
        self.strike = None
        self.type = ''
        self.roughness = 0

        self.polygon_method = None
        self.polygon_alpha = None  # parameter of ashape method
        self.polygon_vertex_ids = None
        self.polygon_vertex_fit_plane = None  # 3d coord of vertex
        self.polygon_centroid_2d = None
        self.polygon_centroid_3d = None
        self.polygon_area = None

        self.trace_type = None
        self.trace_length = None
        self.trace_vertex_1 = None
        self.trace_vertex_2 = None

        self.disc_type = None
        self.disc_center = None
        self.disc_radius = None
        self.disc_normal = self.normal
        self.ellip_a = None
        self.ellip_b = None
        self.extention = 1
        self.long_axis_norm = None
        self.short_axis_norm = None
        self.long_axis_vertex = None
        self.short_axis_vertex = None
        self.ratio_axis = None
        self.calculate_time = 0

        self.block_id = None
        self.block_neighbours = None

    @classmethod
    def from_cluster(cls, cluster_obj: Cluster.Cluster):
        return cls(
            joint_id=cluster_obj.joint_id,
            joint_cluster_id=cluster_obj.joint_cluster_id,
            cluster_id=cluster_obj.cluster_id,
            rock_points=cluster_obj.rock_points,
            plane_params=cluster_obj.plane_params,
            inlier_ratio=cluster_obj.inlier_ratio
        )

    def get_polygon(self, method='convex'):
        '''

        :param method: convex or ashape
        :return: discontinuity
                self.polygon_method = None
                self.polygon_alpha = None  # parameter of ashape method
                self.polygon_vertex_ids = None
                self.polygon_centroid_2d = None
                self.polygon_centroid_3d = None
                self.polygon_area = None
        '''
        self.polygon_method = method
        if method == 'convex':
            self.get_polygon_by_convexhull()
        elif method == 'ashape':
            self.get_polygon_by_ashape()

    def get_polygon_by_convexhull(self):
        """
        获取 cluster.projections 中构成凸包的点的索引。
        :return: 索引列表（按逆时针排列）
        """
        projections = self.projections  # shape (N, 2)

        # 无法构成凸包，直接返回NONE
        if projections.shape[0] < 3 or np.linalg.matrix_rank(projections) < 2:
            self.valid = False
            self.polygon_area = 0.0
            # self.polygon_centroid_2d = np.mean(projections, axis=0)
            # self.polygon_centroid_3d = Discontinuity.project_2d_to_3d(self, self.polygon_centroid_2d)
            return None

        try:
            hull = ConvexHull(projections)
            vertex_indices = hull.vertices.tolist()  # 按右手定则：拇指为法向方向，逆时针顺序#
            hull_pts = projections[vertex_indices]  # 多边形边界点
            self.polygon_vertex_ids = vertex_indices
            self.polygon_vertex_fit_plane = self.project_2d_to_3d(hull_pts)  # shape N,3
            # print(hull_pts)
            # print(self.polygon_vertex_fit_plane)

            # 面积(# 2D 中 volume 即 area)
            if abs(hull.volume) > 1e-8:
                self.valid = True
                self.polygon_area = hull.volume
            else:
                self.valid = False
                self.polygon_area = 0.0
                return None

            # 二维质心：使用多边形质心公式
            x = hull_pts[:, 0]
            y = hull_pts[:, 1]
            area = 0.0
            cx = 0.0
            cy = 0.0
            for i in range(len(hull_pts)):
                j = (i + 1) % len(hull_pts)
                cross = x[i] * y[j] - x[j] * y[i]
                area += cross
                cx += (x[i] + x[j]) * cross
                cy += (y[i] + y[j]) * cross
            area *= 0.5
            if abs(area) < 1e-8:
                centroid = np.mean(hull_pts, axis=0)
            else:
                cx /= (6.0 * area)
                cy /= (6.0 * area)
                centroid = np.array([cx, cy])
            self.polygon_centroid_2d = centroid

            # 三维质心
            self.polygon_centroid_3d = Discontinuity.project_2d_to_3d(self, centroid)

        except:
            # 无法构成凸包，直接返回全部索引
            self.valid = False
            self.polygon_vertex_ids = list(range(projections.shape[0]))
            self.polygon_area = 0.0
            # self.polygon_centroid_2d = np.mean(projections, axis=0)
            # self.polygon_centroid_3d = Discontinuity.project_2d_to_3d(self, self.polygon_centroid_2d)
            return self.polygon_vertex_ids

    def get_polygon_by_ashape(self, alpha=1.0):
        '''
        加速方式：单独并行运算
        获取 cluster.projections 中通过 Alpha Shape 构成边界的点的索引。
        :param alpha: 可选，形状参数；为 None 时自动估计。
        :return: 点索引列表（按逆时针排列）
        '''
        projections = self.projections  # shape (N, 2)

        if projections.shape[0] < 4:
            self.valid = False
            self.polygon_vertex_ids = list(range(projections.shape[0]))
            self.polygon_area = 0.0
            self.polygon_centroid_2d = np.mean(projections, axis=0)
            self.polygon_centroid_3d = self.project_2d_to_3d(self.polygon_centroid_2d)
            return self.polygon_vertex_ids

        # 构建 alpha shape 多边形
        alpha_shape = alphashape.alphashape(projections, alpha)

        # 如果为 MultiPolygon，仅取最大面积那个
        if alpha_shape.geom_type == 'MultiPolygon':
            alpha_shape = max(alpha_shape.geoms, key=lambda p: p.area)

        # 点顺序索引：找到构成边界的点坐标并匹配原始索引
        boundary_coords = np.array(alpha_shape.exterior.coords[:-1])  # 移除重复起点
        vertex_indices = []
        for coord in boundary_coords:
            dists = np.linalg.norm(projections - coord, axis=1)
            idx = np.argmin(dists)
            vertex_indices.append(idx)

        self.polygon_vertex_ids = vertex_indices

        # 返回边界点的三维坐标(保持平面)
        self.polygon_vertex_fit_plane = self.project_2d_to_3d(boundary_coords)

        # 面积
        self.polygon_area = alpha_shape.area
        if self.polygon_area != 0:
            self.valid = True

        # 质心（二维）
        centroid = np.array(alpha_shape.centroid.coords[0])
        self.polygon_centroid_2d = centroid

        # 质心（三维）
        self.polygon_centroid_3d = self.project_2d_to_3d(centroid)
        self.polygon_alpha = alpha

    def get_trace_segment_from_edges(self):
        '''
        从边界点中寻找最远的一对点，作为迹线（trace segment）
        参数:
            polygon_vertexs: 多边形顶点坐标数组，形状为 (N, 3) 的 numpy 数组

        返回:
            最大迹长（浮点数）
        '''
        if len(self.polygon_vertex_ids) < 2 or self.polygon_area == 0:
            raise Exception('No valid edge boundary available to infer trace.')

        # 边界点坐标（二维）
        polygon_vertexs = self.polygon_vertex_fit_plane  # shape(N,3)

        # 计算所有点对的差值
        diff = polygon_vertexs[:, np.newaxis, :] - polygon_vertexs[np.newaxis, :, :]

        # 计算所有点对的距离（平方）
        distances_sq = np.sum(diff ** 2, axis=-1)

        # 获取最大距离（避免开方以节省计算）
        max_distance_sq = np.max(distances_sq)

        # 获取最大距离对应的两个点的索引
        i, j = np.unravel_index(np.argmax(distances_sq), distances_sq.shape)

        # 保存结果
        self.trace_length = np.sqrt(max_distance_sq)
        self.trace_vertex_1 = polygon_vertexs[i]  # shape(3,)=[47.20507861 17.77964028 16.27065507]
        self.trace_vertex_2 = polygon_vertexs[j]
        self.trace_type = 'trace_from_edges'

    def get_trace_segment_from_farthest2(self):
        '''

        :return:
        '''
        projections = self.projections
        dists_to_center = np.linalg.norm(projections - np.asarray([0, 0]), axis=1)
        vertex1_idx = np.argmax(dists_to_center)  # 边缘点1在 projected_2d 中的索引

        dists_to_idx1 = np.linalg.norm(projections - projections[vertex1_idx], axis=1)
        vertex2_idx = np.argmax(dists_to_idx1)  # 边缘点2在 projected_2d 中的索引

        self.trace_length = np.sqrt(np.max(dists_to_idx1))
        self.trace_vertex_1 = self.project_2d_to_3d(projections[vertex1_idx])
        self.trace_vertex_2 = self.project_2d_to_3d(projections[vertex2_idx])
        self.trace_type = 'trace_from_farthest'

    # def get_length(self):
    #     """
    #     根据 trace_vertex_index 中的两个点索引，从 rock_points 中获取坐标并计算三维迹线长度。
    #     """
    #     p1 = self.trace_vertex_1
    #     p2 = self.trace_vertex_2
    #     self.trace_length = np.linalg.norm(p1 - p2)
    #     # 额外的检验
    #     if self.trace_length == 0:
    #         self.valid = False

    def get_disc_circle(self):
        """
        由迹线端点和结构面法向计算圆盘参数，包括中心、法向、半径等。
        """
        self.disc_type = 'circle'
        self.disc_center = self.polygon_centroid_3d
        self.disc_radius = self.trace_length * 0.5
        # self.disc_type = None
        # self.disc_center = None
        # self.disc_radius = None
        # self.disc_normal = self.normal
        # self.ellip_a = None
        # self.ellip_b = None
        # self.long_axis_norm = None
        # self.shrt_axis_norm = None
        # self.long_axis_vertex = None
        # self.short_axis_vertex = None
        # self.ratio_axis = None
        # self.calculate_time = 0

    def get_disc_elliptical(self):
        '''
        在三维空间中构建椭圆的关键点（中心、长轴两端点、短轴两端点）

        参数:
            center (np.ndarray): 椭圆中心点坐标，形状为 (3,)。
            vector_a (np.ndarray): 椭圆长轴方向的单位向量，形状为 (3,)。
            normal (np.ndarray): 椭圆所在平面的法向量（不要求单位长度），形状为 (3,)。
            ellip_a (float): 长半轴长度。
            ellip_b (float): 短半轴长度。

        返回:
            dict: 包含以下键值对的字典：
                - 'center': 椭圆中心点，(3,) array。
                - 'long_axis_endpoints': 长轴两端点，(2,3) array。
                - 'short_axis_endpoints': 短轴两端点，(2,3) array。
        '''
        center = self.polygon_centroid_3d
        vector_a = self.trace_vertex_1 - self.trace_vertex_2
        ellip_a = self.trace_length * 0.5
        ellip_b = self.polygon_area / (np.pi * ellip_a)
        normal = self.normal

        # 归一化长轴方向向量
        dir_a = vector_a / np.linalg.norm(vector_a)

        # 使用法向量和长轴方向构造短轴方向（确保在同一平面内）
        dir_b = np.cross(normal, dir_a)
        norm_b = np.linalg.norm(dir_b)
        if norm_b < 1e-8:
            raise ValueError("法向量与长轴方向平行，无法构造椭圆面")
        dir_b = dir_b / norm_b

        # 计算长轴两端点
        p_a1 = center + ellip_a * dir_a
        p_a2 = center - ellip_a * dir_a

        # 计算短轴两端点
        p_b1 = center + ellip_b * dir_b
        p_b2 = center - ellip_b * dir_b

        # return elliptical parameters
        self.disc_type = 'elliptical'
        self.disc_center = center
        self.disc_radius = None
        self.ellip_a = ellip_a * self.extention
        self.ellip_b = ellip_b * self.extention
        self.long_axis_norm = dir_a
        self.short_axis_norm = dir_b
        self.long_axis_vertex = np.asarray([p_a1, p_a2])
        self.short_axis_vertex = np.asarray([p_b1, p_b2])
        self.ratio_axis = self.ellip_b / self.ellip_a

    def get_roughness(self, method='pca'):
        '''
        加速方式：单独并行运算
        :param method:
        :return:
        '''
        np_points = np.asarray([point.coord for point in self.rock_points.points])
        if method == 'pca':
            pca = PCA(n_components=3)
            pca.fit(np_points)
            normal = pca.components_[-1]
            residuals = (np_points - np_points.mean(0)) @ normal
            self.roughness = np.std(residuals)
        elif method == 'zrange':
            z_proj = np_points @ self.normal
            self.roughness = np.max(z_proj) - np.min(z_proj)

    def visualize(self, plotter=None, show=True):
        if plotter is None:
            plotter = pv.Plotter()
        np_points = np.asarray([point.coord for point in self.rock_points.points])
        plotter.add_points(np_points, color='blue', point_size=5)

        if self.ashape_vertex is not None:
            boundary = pv.lines_from_points(self.ashape_vertex, close=True)
            plotter.add_mesh(boundary, color='green', line_width=3)

        if self.disk_radius is not None:
            circle = pv.Circle(radius=self.disk_radius, resolution=100)
            circle.rotate_vector(self.normal, 0)
            circle.translate(self.disk_centroid)
            plotter.add_mesh(circle, color='red', opacity=0.5)

        if show:
            plotter.show()

    def print_all_attributes(self):
        """
        打印当前对象及其父类的所有属性及其对应值。
        """
        print(f"Attributes of {self.__class__.__name__}:")
        # 用集合去重
        printed_keys = set()
        for cls in self.__class__.__mro__:
            if cls is object:
                continue
            print(f"\nFrom class: {cls.__name__}")
            for key, value in cls.__dict__.items():
                # 跳过函数和类方法等
                if callable(value) or key.startswith('__'):
                    continue
            for key, value in self.__dict__.items():
                if key not in printed_keys:
                    print(f"  {key} = {value}")
                    printed_keys.add(key)


class Discontinuitys:
    def __init__(self):
        self.discontinuitys = []
        self.where = 'save_path'

    def add(self, disc: Discontinuity):
        self.discontinuitys.append(disc)

    def delete(self, cluster_id_list):
        self.discontinuitys = [dis for dis in self.discontinuitys if dis.cluster_id not in cluster_id_list]

    def load_database(self, path):
        with open(path, 'rb') as f:
            self.discontinuitys = pickle.load(f)

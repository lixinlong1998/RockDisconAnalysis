import pickle
import numpy as np
from Cluster import Cluster
import numpy as np
from scipy.optimize import leastsq
from sklearn.decomposition import PCA
import pyvista as pv
from shapely.geometry import MultiPoint
from shapely.ops import triangulate
from scipy.spatial import Delaunay


class Rockblock():
    def __init__(self, block_id, vertexies, edges, lithology, density, volume, ...):
        self.block_id = block_id
        self.neighbours = {block_id: [discontinuity_id, ...], ...}  # 该block邻接的block_id以及对应的共用discontinuitys的id

        self.vertexies = vertexies  # 顶点序列
        self.edges = edges  # 顶点对
        self.centroid_gravity = None  # 默认为多面体的几何中心
        self.centroid_geometry = None  # 多面体的几何中心
        self.discontinuitys = None  # discontinuity列表

        self.face_number = None  # 多面体的表面数量
        self.height = None  # 外接矩形高
        self.width = None  # 外接矩形宽
        self.depth = None  # 外接矩形深
        self.volume = None  # 多面体打散成四面体，对所有四面体体积求和

        self.lithology = lithology  # 给定
        self.density = density  # 给定
        self.weight = self.density * self.volume

        self.stability_type = None  # 基于块体理论分析岩块的活动性
        self.failure_type = None  # 分析块体主控面和失稳模式
        self.rock_quality = None  # 给定（表面节理裂隙密度）
        self.non_persistent = None  # 块体中的非断裂性质
        self.dangerous_index = None  # 岩块的危险指数

    def get_alpha_shape(self, alpha=1.0):
        '''
        基于投影点计算 alpha shape 边界，并返回对应的三维边界点
        '''
        proj_2d = self.projections
        tri = Delaunay(proj_2d)
        triangles = proj_2d[tri.simplices]

        def edge_length_squared(p1, p2):
            return np.sum((p1 - p2) ** 2)

        def is_triangle_in_alpha(tri):
            a, b, c = tri
            s1 = edge_length_squared(a, b)
            s2 = edge_length_squared(b, c)
            s3 = edge_length_squared(c, a)
            s = (np.sqrt(s1) + np.sqrt(s2) + np.sqrt(s3)) / 2
            area = np.sqrt(s * (s - np.sqrt(s1)) * (s - np.sqrt(s2)) * (s - np.sqrt(s3)))
            if area == 0:
                return False
            radius = (np.sqrt(s1) * np.sqrt(s2) * np.sqrt(s3)) / (4.0 * area)
            return radius < (1.0 / alpha)

        edges = set()
        for t in triangles:
            if is_triangle_in_alpha(t):
                for i in range(3):
                    edge = tuple(sorted((tuple(t[i]), tuple(t[(i + 1) % 3]))))
                    edges.add(edge)

        boundary_pts = list(set([p for e in edges for p in e]))
        boundary_pts = np.array(boundary_pts)

        u, v = self.projections_uv
        uv2xyz = lambda uv: self.centroid + uv[0] * u + uv[1] * v
        self.ashape_vertex = np.array([uv2xyz(pt) for pt in boundary_pts])
        self.polygon_area = self.compute_polygon_area(boundary_pts)
        self.ashape_alpha = alpha
        self.ashape_centroid = np.mean(self.ashape_vertex, axis=0)

    def get_face_area(self, poly_2d):
        x, y = poly_2d[:, 0], poly_2d[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def get_disc_circle(self):
        proj_2d = self.projections

        def calc_R(xc, yc):
            return np.sqrt((proj_2d[:, 0] - xc) ** 2 + (proj_2d[:, 1] - yc) ** 2)

        def f(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = np.mean(proj_2d, axis=0)
        center, _ = leastsq(f, center_estimate)
        Ri = calc_R(*center)
        radius = Ri.mean()

        u, v = self.projections_uv
        self.disk_centroid = self.centroid + center[0] * u + center[1] * v
        self.disk_radius = radius
        self.disk_normal = self.normal

    def get_disc_elliptical(self):
        pts = self.points.points
        if len(pts) < 3:
            return np.zeros(3), np.zeros(3), 0.0

        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        sort_idx = np.argsort(eigvals)[::-1]  # 降序排列
        v1 = eigvecs[:, sort_idx[0]]
        v2 = eigvecs[:, sort_idx[1]]
        p1 = self.centroid + v1 * np.std(pts @ v1)
        p2 = self.centroid - v1 * np.std(pts @ v1)
        p3 = self.centroid + v2 * np.std(pts @ v2)
        p4 = self.centroid - v2 * np.std(pts @ v2)
        long_axis = p1 if np.linalg.norm(p1 - p2) > np.linalg.norm(p3 - p4) else p3
        short_axis = p3 if np.linalg.norm(p1 - p2) > np.linalg.norm(p3 - p4) else p1
        ratio = np.linalg.norm(p1 - p2) / np.linalg.norm(p3 - p4 + 1e-6)
        self.long_axis = long_axis
        self.short_axis = short_axis
        self.ratio = ratio

    def get_roughness(self, method='pca'):
        coords = np.array(self.points)
        if method == 'pca':
            pca = PCA(n_components=3)
            pca.fit(coords)
            normal = pca.components_[-1]
            residuals = (coords - coords.mean(0)) @ normal
            self.roughness = np.std(residuals)
        elif method == 'zrange':
            z_proj = coords @ self.normal
            self.roughness = np.max(z_proj) - np.min(z_proj)

    def visualize(self, plotter=None, show=True):
        if plotter is None:
            plotter = pv.Plotter()
        points = np.array([p.coord for p in self.points.points])
        plotter.add_points(points, color='blue', point_size=5)

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


class Discontinuitys:
    def __init__(self):
        self.discontinuitys = []
        self.where = 'save_path'

    def add(self, disc: Discontinuity):
        self.discontinuitys.append(disc)

    def delete(self, cluster_id_list):
        self.discontinuitys = [dis for dis in self.discontinuitys if dis.cluster_id not in cluster_id_list]

    def export(self, path, format='pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.discontinuitys, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.discontinuitys = pickle.load(f)

import numpy as np
from scipy.spatial import ConvexHull
import numpy as np
import pyvista as pv
import tetgen


class Rockblock:
    def __init__(self, block_id, vertexies, faces, lithology, density, discontinuitys=None, neighbours=None):
        self.block_id = block_id
        self.vertexies = np.array(vertexies)  # 顶点序列，(N, 3)
        self.faces = np.array(faces)  # [[vertex_id,...],...]
        self.lithology = lithology
        self.density = density
        self.discontinuitys = discontinuitys if discontinuitys else []
        self.neighbours = neighbours  # dict(){other_block_id: [discontinuity_ids]}

        self.face_number = self.compute_face_number()
        self.height, self.width, self.depth = self.compute_bounding_box_dimensions()
        self.edges = self.get_edges()  # 顶点对，例如 [(0,1), (1,2), ...]
        self.mesh = self.triangulate_faces_and_build_mesh()
        self.tet_points, self.tet_elements, self.tet_centroids, self.tet_volumes = self.tetrahedralize()

        self.centroid_geometry = self.compute_geometric_centroid()
        self.centroid_gravity = self.compute_gravity_centroid()
        self.volume = self.compute_volume()

        self.weight = self.density * self.volume if self.volume else 0

        # Rock mechanical characterization
        self.stability_type = None
        self.failure_type = None
        self.rock_quality = None
        self.non_persistent = None
        self.dangerous_index = None

    def get_edges(self):
        """
        获取所有不重复的边：每条边是一个顶点索引对 (i, j)，要求 i < j。
        """
        faces = np.array(self.faces)
        v1 = faces
        v2 = np.roll(faces, shift=-1, axis=1)
        edges = np.stack([np.minimum(v1, v2), np.maximum(v1, v2)], axis=2)
        edges = edges.reshape(-1, 2)
        edges = np.unique(edges, axis=0)
        return edges

    def triangulate_faces_and_build_mesh(self):
        """
        输入：多面体面片的顶点索引列表（每个元素是一个面，面由顶点索引列表表示）
        输出：构造 PyVista PolyData 并赋值给 self.mesh
        """
        points = self.vertexies
        all_tri_faces = []

        for face in self.faces:
            if len(face) < 3:
                continue  # 忽略无效面
            elif len(face) == 3:
                all_tri_faces.append([3, *face])
            else:
                # 三角扇分解，例如 [0,1,2,3] -> (0,1,2), (0,2,3)
                for i in range(1, len(face) - 1):
                    tri = [face[0], face[i], face[i + 1]]
                    all_tri_faces.append([3, *tri])

        # 构造 PolyData
        '''
        PyVista 要求面的输入是一个一维数组，结构如下：
        [ n0, i0_0, i0_1, ..., i0_n0, n1, i1_0, i1_1, ..., i1_n1, ... ]
            n0: 第一个面的顶点个数
            i0_0, i0_1, ..., i0_n0: 第一个面的顶点索引
            n1: 第二个面的顶点个数
            ...以此类推
        '''
        face_array = np.hstack(all_tri_faces)  # 而 PyVista 需要的是一个一维列表/数组，所以用 np.hstack 将其“拍平”。
        mesh = pv.PolyData(points, face_array)
        return mesh

    def tetrahedralize(self):
        # 用 TetGen 对 PolyData 进行四面体剖分
        tgen = tetgen.TetGen(self.mesh)
        tgen.tetrahedralize()
        tet_mesh = tgen.grid
        tet_points = tet_mesh.points  # (N, 3) array, 所有顶点的坐标

        # 提取四面体索引
        try:
            tet_elements = tet_mesh.cells_dict[pv.CellType.TETRA]  # (M, 4) array，每行表示一个四面体由哪4个点索引组成
        except AttributeError:
            raise RuntimeError("未生成四面体单元，请确认输入 surface 是闭合多面体。")

        # 从顶点索引中提取四面体四个顶点坐标（向量化）
        vertex_1 = tet_points[tet_elements[:, 0]]  # (M, 3)
        vertex_2 = tet_points[tet_elements[:, 1]]  # (M, 3)
        vertex_3 = tet_points[tet_elements[:, 2]]  # (M, 3)
        vertex_4 = tet_points[tet_elements[:, 3]]  # (M, 3)

        # 计算每个四面体的重心（centroid）
        tet_centroids = (vertex_1 + vertex_2 + vertex_3 + vertex_4) / 4.0  # (M, 3)

        # 计算每个四面体的体积（volume）
        v1 = vertex_2 - vertex_1
        v2 = vertex_3 - vertex_1
        v3 = vertex_4 - vertex_1
        cross = np.cross(v2, v3)  # (M, 3)
        dot = np.einsum('ij,ij->i', v1, cross)  # (M,), 点积
        tet_volumes = np.abs(dot) / 6.0  # (M,)

        # 返回：
        # - tet_points: (N, 3) array, 所有顶点坐标
        # - tet_elements: (M, 4) array, 每行是一个四面体由四个点构成的索引
        # - tet_centroids: (M, 3) array, 每个四面体的几何重心坐标
        # - tet_volumes: (M,) array, 每个四面体的体积
        return tet_points, tet_elements, tet_centroids, tet_volumes

    def show_tetrahedrons(self):
        # 可视化
        p = pv.Plotter()
        p.add_mesh(self.mesh, show_edges=True, show_scalar_bar=False, opacity=0.5)
        p.show()

        # 在主函数结尾添加此部分代码
        # 单独显示每个四面体
        colors = [
            'red', 'green', 'blue', 'cyan', 'magenta',
            'yellow', 'orange', 'purple', 'brown', 'gray'
        ]
        p = pv.Plotter()
        for i, tet in enumerate(self.tet_elements):
            tet_pts_local = self.tet_points[tet]
            # 每个四面体由 4 个顶点定义
            cells = np.hstack([[4, 0, 1, 2, 3]])
            local_grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], tet_pts_local)
            p.add_mesh(local_grid, show_edges=True, opacity=0.7, color=colors[i % len(colors)])
        p.show()

    def compute_geometric_centroid(self):
        """
        计算几何质心，即所有顶点的平均值。
        """
        if len(self.vertexies) == 0:
            return np.zeros(3)
        return np.mean(self.vertexies, axis=0)

    def compute_gravity_centroid(self):
        """
        如果已实现四面体分解，则重心为四面体重心之平均。
        否则与当前几何中心相同，预留接口。
        """
        if self.tet_centroids != None:
            return np.mean(self.tet_centroids, axis=0)
        else:
            return self.compute_geometric_centroid()

    def compute_face_number(self):
        """
        用于后续基于面片列表获取实际数量。当前假设每个不重复的边界面为一面。
        若使用 PolyData 或面定义方法，应替换此方法。
        """
        if self.discontinuitys:
            return len(self.discontinuitys)
        else:
            # 仅基于边推断面数困难，保守返回 None
            return None

    def compute_bounding_box_dimensions(self):
        """
        计算包围盒的尺寸（height, width, depth）
        """
        if self.vertexies.shape[0] == 0:
            return 0, 0, 0
        min_corner = np.min(self.vertexies, axis=0)
        max_corner = np.max(self.vertexies, axis=0)
        dims = max_corner - min_corner
        return dims[2], dims[0], dims[1]  # 高-宽-深

    def compute_volume(self):
        """
        使用 ConvexHull 将点构造成三维多面体，并计算体积（保守估计）。
        若为凹多面体应使用更精细的分解算法（如四面体剖分）。
        """
        try:
            if len(self.vertexies) < 4:
                return 0
            hull = ConvexHull(self.vertexies)
            return hull.volume
        except:
            return 0

    def estimate_stability(self, gravity=np.array([0, 0, -1])):
        """
        简化判别：若存在一个结构面法向朝上，且与重心连线方向接近，则可能失稳。
        """
        if not self.discontinuitys or not self.centroid_gravity.any():
            return False

        for disc in self.discontinuitys:
            n = disc.normal
            p = disc.centroid
            vec = self.centroid_gravity - p
            vec = vec / np.linalg.norm(vec)
            # 若重心连线与法向夹角小于某个阈值，表示重力“指向”该结构面，可能滑出
            angle = np.arccos(np.clip(np.dot(vec, n), -1, 1)) * 180 / np.pi
            if angle < 45:
                self.stability_type = "unstable"
                return False

        self.stability_type = "stable"
        return True

    def classify_failure(self):
        """
        计算重心到所有结构面的矢量夹角，判断主控面是否向自由面倾斜
        """
        if not self.discontinuitys or not self.centroid_gravity.any():
            return "unknown"

        centroid = self.centroid_gravity
        labels = []

        for disc in self.discontinuitys:
            n = disc.normal  # 应该为单位法向量
            p = disc.centroid  # 面的重心点
            vec = centroid - p
            vec = vec / np.linalg.norm(vec)
            angle = np.arccos(np.clip(np.dot(vec, n), -1, 1)) * 180 / np.pi  # 单位：度

            if angle < 30:
                labels.append('滑移')
            elif 30 <= angle < 60:
                labels.append('倾倒')
            else:
                labels.append('落石')

        from collections import Counter
        most_common = Counter(labels).most_common(1)
        self.failure_type = most_common[0][0] if most_common else 'unknown'
        return self.failure_type

    def compute_dangerous_index(self, alpha=1.0, beta=0.5, gamma=0.3):
        """
        危险指数 = α * 体积 + β * 结构面数 + γ * 自由度（近似用边数 - 面数估算）
        所有项归一化后进行加权求和。
        """
        volume = self.volume
        face_count = self.face_number or len(self.faces)
        edge_count = len(self.edges)

        # 粗略估计自由度为边数-面数（未考虑约束性）
        freedom = max(edge_count - face_count, 1)

        # 归一化（假设最大值用于测试）
        volume_norm = volume / 1000.0
        face_norm = face_count / 20.0
        freedom_norm = freedom / 30.0

        self.dangerous_index = alpha * volume_norm + beta * face_norm + gamma * freedom_norm
        return self.dangerous_index

    def __repr__(self):
        return f"<Rockblock id={self.block_id}, faces={self.face_number}, volume={self.volume:.2f}, weight={self.weight:.2f}>"





# class Vertex:
#     def __init__(self, X: float, Y: float, Z: float, point_id: int, joint_id: int, joint_cluster_id: int,
#                  cluster_id: int, R, G, B, a, b, c, d):
#         self.valid = True
#         self.coord = np.asarray([X, Y, Z], dtype=np.float64)
#         self.point_id = point_id
#         self.joint_id = joint_id
#         self.joint_cluster_id = joint_cluster_id
#         self.cluster_key = (self.joint_id, self.joint_cluster_id)
#         self.cluster_id = cluster_id
#         self.color = np.asarray([R, G, B], dtype=np.uint8)
#         self.plane_paras = np.asarray([a, b, c, d], dtype=np.float64)


# class Edge:
#     def __init__(self, X: float, Y: float, Z: float, point_id: int, joint_id: int, joint_cluster_id: int,
#                  cluster_id: int, R, G, B, a, b, c, d):
#         self.valid = True
#         self.start_vertex = pt_start
#         self.start_vertex = pt_start
#         self.coord = np.asarray([X, Y, Z], dtype=np.float64)
#         self.point_id = point_id
#         self.joint_id = joint_id
#         self.joint_cluster_id = joint_cluster_id
#         self.cluster_key = (self.joint_id, self.joint_cluster_id)
#         self.cluster_id = cluster_id
#         self.color = np.asarray([R, G, B], dtype=np.uint8)
#         self.plane_paras = np.asarray([a, b, c, d], dtype=np.float64)


# class Polygon:
#     def __init__(self, X: float, Y: float, Z: float, point_id: int, joint_id: int, joint_cluster_id: int,
#                  cluster_id: int, R, G, B, a, b, c, d):
#         self.valid = True
#         self.start_vertex = pt_start
#         self.start_vertex = pt_start
#         self.coord = np.asarray([X, Y, Z], dtype=np.float64)
#         self.point_id = point_id
#         self.joint_id = joint_id
#         self.joint_cluster_id = joint_cluster_id
#         self.cluster_key = (self.joint_id, self.joint_cluster_id)
#         self.cluster_id = cluster_id
#         self.color = np.asarray([R, G, B], dtype=np.uint8)
#         self.plane_paras = np.asarray([a, b, c, d], dtype=np.float64)

class Face:
    """
    表示由结构面交线组成的一个封闭面（loop），用于构建块体。
    """

    def __init__(self, node_indices, surface_id, node_coords):
        """
        :param node_indices: List[int]，构成面片的节点索引（需按顺时针或逆时针排序）
        :param surface_id: Tuple[int, int]，构成该面的结构面ID（如 cluster_id）
        :param node_coords: np.ndarray, shape=(N, 3)，所有节点的三维坐标集合（用于计算几何属性）
        """
        self.node_indices = node_indices
        self.surface_id = surface_id
        self.coords = node_coords[node_indices]  # shape=(n, 3)

        # 计算属性
        self.centroid = self.compute_centroid()
        self.normal = self.compute_normal()
        self.area = self.compute_area()

    def compute_centroid(self):
        """ 计算面片重心 """
        return np.mean(self.coords, axis=0)

    def compute_normal(self):
        """
        利用多边形法向量计算公式，适用于任意共面多边形
        """
        n = np.zeros(3)
        for i in range(len(self.coords)):
            p1 = self.coords[i]
            p2 = self.coords[(i + 1) % len(self.coords)]
            n += np.cross(p1, p2)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-8 else np.array([0, 0, 0])

    def compute_area(self):
        """
        使用三角剖分的方法计算任意多边形面积（假设共面）
        """
        if len(self.coords) < 3:
            return 0.0
        ref = self.coords[0]
        area = 0.0
        for i in range(1, len(self.coords) - 1):
            v1 = self.coords[i] - ref
            v2 = self.coords[i + 1] - ref
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))
        return area

    def is_adjacent_to(self, other_face, tol=1e-6):
        """
        判断是否与另一个面邻接：共享2个以上节点可视为邻接（容差内）
        """
        this_set = set(self.node_indices)
        other_set = set(other_face.node_indices)
        return len(this_set & other_set) >= 2

    def is_valid(self, area_tol=1e-6):
        """ 判断面片是否合法：面积大于阈值、法向量非零 """
        return self.area > area_tol and np.linalg.norm(self.normal) > 1e-6

    def reverse(self):
        """ 翻转面片顶点顺序（翻转法向） """
        self.node_indices = self.node_indices[::-1]
        self.coords = self.coords[::-1]
        self.normal = -self.normal

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
        self.valid = True
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
        self.where = 'save_path'

    @property
    def number(self):
        return len(self.points)

    def append(self, Point3D):
        self.points.append(Point3D)

    def add(self, X: float, Y: float, Z: float, point_id: int, joint_id: int, joint_cluster_id: int,
            cluster_id: int, R, G, B, a, b, c, d):
        self.points.append(Point3D(X, Y, Z, point_id, joint_id, joint_cluster_id, cluster_id, R, G, B, a, b, c, d))

    def delete(self, point_id_list):
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

    def export(self, path, format='ply_bin'):
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

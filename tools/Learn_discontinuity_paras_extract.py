import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import trimesh
import pandas as pd
import time

# 0. 开始总计时
start_total = time.time()

# 1. 读取点云与估计法向
start = time.time()
pcd = o3d.io.read_point_cloud("icosahedron_noisy.ply")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(k=30)
# 如果法向与点到质心方向夹角为钝角，则取反（使其朝外）
center = np.mean(np.asarray(pcd.points), axis=0)
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
vectors_to_center = center - points
cos_angles = np.einsum("ij,ij->i", normals, vectors_to_center)
normals[cos_angles > 0] *= -1  # 使法向朝外

pcd.normals = o3d.utility.Vector3dVector(normals)
print(f"1. 读取与法线估计耗时：{time.time() - start:.2f}s")

# 2. 使用 DBSCAN 聚类寻找平面片段
start = time.time()
d = np.einsum('ij,ij->i', normals, points)  # 每点的平面偏移量
features = np.concatenate([normals, d[:, None]], axis=1)
db = DBSCAN(eps=0.01, min_samples=50).fit(features)
clusters = db.labels_
unique_clusters = np.unique(clusters[clusters >= 0])
print(f"2. cluster 聚类耗时：{time.time() - start:.2f}s")
print(f"   有效聚类数: {len(unique_clusters)}")

# 3. 基于平面法向夹角分组 set（小于30度视为同组）
start = time.time()
cluster_normals = []
for c in unique_clusters:
    idx = np.where(clusters == c)[0]
    cluster_normals.append(normals[idx].mean(axis=0))
cluster_normals = np.array([v / np.linalg.norm(v) for v in cluster_normals])

sets = -np.ones(len(cluster_normals), dtype=int)
set_id = 0
for i in range(len(cluster_normals)):
    if sets[i] != -1:
        continue
    sets[i] = set_id
    for j in range(i + 1, len(cluster_normals)):
        angle = np.degrees(np.arccos(np.clip(np.dot(cluster_normals[i], cluster_normals[j]), -1.0, 1.0)))
        if angle < 30:
            sets[j] = set_id
    set_id += 1
print(f"3. set 倾向-倾角分组耗时：{time.time() - start:.2f}s")
print(f"   平面组数量: {set_id}")

# 4. 可视化标注
start = time.time()
colors = np.zeros((points.shape[0], 3))
colormap = np.random.rand(len(unique_clusters), 3)
if len(unique_clusters) > 0:
    for i, c in enumerate(unique_clusters):
        colors[clusters == c] = colormap[i]
else:
    print("⚠️ 未找到有效聚类！请检查 DBSCAN 参数或输入数据。")
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
print(f"4. 可视化耗时：{time.time() - start:.2f}s")

# 5. 拟合每个聚类的平面，提取几何属性
start = time.time()
planes = []
discs = []
for i, c in enumerate(unique_clusters):
    idx = np.where(clusters == c)[0]
    pts = points[idx]
    centroid = pts.mean(axis=0)
    uu, dd, vv = np.linalg.svd(pts - centroid)
    normal = vv[-1]
    A, B, C = normal
    D = -normal.dot(centroid)
    dip = np.degrees(np.arccos(abs(C)))
    dip_dir = (np.degrees(np.arctan2(B, A)) + 360) % 360
    try:
        line = LineString([(-10, (-A * -10 - D) / B), (10, (-A * 10 - D) / B)])
        hull = Polygon(pts[:, :2]).convex_hull
        segment = hull.intersection(line)
        if segment.geom_type == 'LineString':
            x0, y0 = segment.coords[0]
            x1, y1 = segment.coords[-1]
            length = segment.length
            midpoint = ((x0 + x1) / 2, (y0 + y1) / 2, 0)
            radius = length / 2
            circle2d = Point(midpoint[:2]).buffer(radius, resolution=64)
            mesh2d = trimesh.creation.extrude_polygon(circle2d, height=0.001)
            mesh2d.apply_translation([0, 0, midpoint[2]])
            discs.append(mesh2d)
            planes.append({
                "cluster": int(c),
                "set": int(sets[i]),
                "A,B,C,D": (A, B, C, D),
                "dip": dip,
                "dip_direction": dip_dir,
                "trace_endpoints": ((x0, y0, 0), (x1, y1, 0)),
                "trace_length": length,
                "num_points": len(idx)
            })
    except Exception as e:
        continue
print(f"5. 平面方程与几何特征提取耗时：{time.time() - start:.2f}s")

# 6. 生成交集多面体
start = time.time()
if len(discs) >= 2:
    shapely_polys = [Polygon(d.vertices[:, :2]).convex_hull for d in discs]
    intersect2d = shapely_polys[0]
    for p in shapely_polys[1:]:
        intersect2d = intersect2d.intersection(p)
    zs = [mesh.vertices[:, 2].mean() for mesh in discs]
    min_z, max_z = min(zs), max(zs)
    poly_mesh = trimesh.creation.extrude_polygon(intersect2d, height=max_z - min_z)
    poly_mesh.apply_translation([0, 0, min_z])
    poly_mesh.export("intersection_polyhedron.ply")
    poly_mesh.export("intersection_polyhedron.obj")
    print(f"6. 多面体生成与导出耗时：{time.time() - start:.2f}s")
else:
    print("⚠️ 有效圆盘不足，无法生成交集多面体")

# 7. 导出参数 CSV
start = time.time()
records = []
for p in planes:
    (x0, y0, z0), (x1, y1, z1) = p["trace_endpoints"]
    record = {
        "cluster": p["cluster"],
        "set": p["set"],
        "num_points": p["num_points"],
        "A": p["A,B,C,D"][0],
        "B": p["A,B,C,D"][1],
        "C": p["A,B,C,D"][2],
        "D": p["A,B,C,D"][3],
        "dip": p["dip"],
        "dip_direction": p["dip_direction"],
        "trace_length": p["trace_length"],
        "S_X": x0, "S_Y": y0, "S_Z": z0,
        "E_X": x1, "E_Y": y1, "E_Z": z1
    }
    records.append(record)
df = pd.DataFrame(records)
df.to_csv("icosahedron_plane_clusters.csv", index=False)
print(f"7. 参数保存耗时：{time.time() - start:.2f}s")

# 总耗时
print(f"✅ 总耗时：{time.time() - start_total:.2f}s")

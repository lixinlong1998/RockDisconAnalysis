import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import alphashape

# 生成带有凹边的星形点云
def generate_star_points(num_points=100000, inner_radius=0.5, outer_radius=1.0):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.empty(num_points)
    radii[::2] = outer_radius
    radii[1::2] = inner_radius
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.vstack((x, y)).T
    noise = 0.05 * np.random.randn(*points.shape)
    return points + noise

points = generate_star_points()

# 构建凸包与 alpha shape
alpha = 1
starttime = time.perf_counter()
hull = ConvexHull(points)
print(f'[time cost]{time.perf_counter() - starttime} — calculate ConvexHull .')

starttime = time.perf_counter()
alpha_shape = alphashape.alphashape(points, alpha)
print(f'[time cost]{time.perf_counter() - starttime} — calculate alpha_shape .')


# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(points[:, 0], points[:, 1], 'o', markersize=4, label='Points')

# 凸包绘制
for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=1, label='Convex Hull' if simplex[0] == hull.simplices[0][0] else "")

# α-shape 绘制（替代 PolygonPatch）
if isinstance(alpha_shape, Polygon):
    x, y = alpha_shape.exterior.xy
    polygon = MplPolygon(np.column_stack((x, y)), closed=True, facecolor='red', edgecolor='red', alpha=0.4, label='Alpha Shape')
    ax.add_patch(polygon)

ax.set_aspect('equal')
ax.legend()
ax.set_title('Comparison of Convex Hull and Alpha Shape (Concave)')
plt.show()

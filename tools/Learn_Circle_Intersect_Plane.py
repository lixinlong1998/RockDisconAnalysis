import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 示例数据
C = np.array([1, 2, 3])  # 圆心坐标
n = np.array([1, 1, 1])  # 圆法向量
r = 2                   # 半径
plane = [1, -1, 1, -2]  # 平面 Ax + By + Cz + D = 0，其中 A=1, B=-1, C=1, D=-2

# 单位法向量
n_hat = n / np.linalg.norm(n)

# 构造与 n_hat 垂直的单位向量 u
arbitrary = np.array([0, 0, 1]) if not np.allclose(n_hat, [0, 0, 1]) else np.array([1, 0, 0])
u = np.cross(n_hat, arbitrary)
u /= np.linalg.norm(u)

# 构造另一个垂直向量 v
v = np.cross(n_hat, u)

# 平面参数
A, B, Cc, D = plane

# 计算三角系数
alpha = A * u[0] + B * u[1] + Cc * u[2]
beta = A * v[0] + B * v[1] + Cc * v[2]
gamma = A * C[0] + B * C[1] + Cc * C[2] + D
R = np.sqrt(alpha**2 + beta**2)

# 检查是否有交点
if abs(gamma / (r * R)) > 1:
    intersections = []
else:
    phi = np.arctan2(beta, alpha)
    theta1 = phi + np.arccos(-gamma / (r * R))
    theta2 = phi - np.arccos(-gamma / (r * R))

    # 交点
    P1 = C + r * np.cos(theta1) * u + r * np.sin(theta1) * v
    P2 = C + r * np.cos(theta2) * u + r * np.sin(theta2) * v
    intersections = [P1, P2]

# 画图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制圆
theta = np.linspace(0, 2 * np.pi, 100)
circle_points = np.array([C + r * np.cos(t) * u + r * np.sin(t) * v for t in theta])
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'b', label='Circle')

# 绘制平面
xx, yy = np.meshgrid(np.linspace(-2, 5, 10), np.linspace(-2, 5, 10))
zz = (-A * xx - B * yy - D) / Cc
ax.plot_surface(xx, yy, zz, alpha=0.5, color='orange', label='Plane')

# 绘制交点
if intersections:
    ax.scatter(*P1, color='r', s=50, label='Intersection 1')
    ax.scatter(*P2, color='g', s=50, label='Intersection 2')

# 设置图形
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Intersection of 3D Circle and Plane')
ax.legend()
plt.tight_layout()
plt.show()

# 返回交点数据
intersections

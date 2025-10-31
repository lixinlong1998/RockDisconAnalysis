import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

# 修改后的几何处理函数 -------------------------------------------------
def is_coplanar(points, tolerance=1e-6):
    """
    检查一组点是否共面
    :param points: 三维点列表，形状为[N,3]
    :param tolerance: 浮点容差
    :return: 是否共面 (布尔值)
    """
    if len(points) <= 3:
        return True  # 三点一定共面

    # 取前三个点计算平面方程
    p0, p1, p2 = points[:3]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)

    # 平面方程: ax + by + cz + d = 0
    a, b, c = normal
    d = -np.dot(normal, p0)

    # 检查所有点是否满足平面方程
    for p in points[3:]:
        dist = abs(a * p[0] + b * p[1] + c * p[2] + d) / np.linalg.norm(normal)
        if dist > tolerance:
            return False
    return True


def normalize_ring(ring):
    """
    标准化环路表示（选择最小顶点为起点，并统一方向）
    :param ring: 顶点索引列表
    :return: 标准化后的环路元组
    """
    # 找到最小顶点的位置
    min_idx = np.argmin(ring)
    # 重新排列环路
    normalized = ring[min_idx:] + ring[:min_idx]
    # 统一方向（例如按顺时针排列）
    if normalized[1] > normalized[-1]:
        normalized = normalized[::-1]
        normalized = [normalized[-1]] + normalized[:-1]
    return tuple(normalized)


def extract_polyhedral_blocks(graph, max_cycle_length=8):
    """
    从交线网络中提取多面体块体
    :param graph: NetworkX无向图，节点为顶点坐标，边为连接关系
           max_cycle_length: 允许的最大边数（防止计算爆炸）
    :return: 共面闭合环列表
    """
    # 步骤1: 查找所有简单闭合环
    directed_graph = graph.to_directed()

    # 使用限制长度的环搜索
    cycles = []
    for cycle in nx.simple_cycles(directed_graph):
        if 3 <= len(cycle) <= max_cycle_length:
            cycles.append(cycle)

    # 共面性检测和去重
    valid_faces = set()
    for cycle in cycles:
        points = [graph.nodes[n]['pos'] for n in cycle]
        if is_coplanar(np.array(points)):
            valid_faces.add(normalize_ring(cycle))

    return list(valid_faces)


# 随机三维网络生成器 -------------------------------------------------
def generate_random_3d_graph(num_nodes=15,
                             edge_prob=0.15,
                             seed=None,
                             bbox_size=2.0):
    """
    生成随机三维连通线框图，但是不存在某条线穿过环的情况
    :param num_nodes: 顶点数量 (至少4个)
    :param edge_prob: 边生成概率（控制连接密度）
    :param seed: 随机种子
    :param bbox_size: 空间包围盒尺寸
    :return: NetworkX图对象
    """
    np.random.seed(seed)
    G = nx.Graph()

    # 生成随机三维顶点
    nodes = np.random.uniform(-bbox_size / 2, bbox_size / 2,
                              (num_nodes, 3))
    for i, pos in enumerate(nodes):
        G.add_node(i, pos=pos)

    # 确保连通性的最小生成树
    dist_matrix = np.sqrt(((nodes[:, np.newaxis] - nodes) ** 2).sum(axis=2))
    mst = nx.minimum_spanning_tree(nx.from_numpy_array(dist_matrix))

    # 添加随机边（包含MST基础边）
    for u, v in mst.edges():
        G.add_edge(u, v)

    # 添加额外随机边
    for u, v in combinations(range(num_nodes), 2):
        if u != v and np.random.rand() < edge_prob:
            G.add_edge(u, v)

    return G


def create_cube_graph():
    """创建立方体图结构"""
    G = nx.Graph()
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, pos in enumerate(vertices):
        G.add_node(i, pos=np.array(pos))
    G.add_edges_from(edges)
    return G


def create_pyramid_graph():
    """创建金字塔图结构"""
    G = nx.Graph()
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # 底面
        (0.5, 0.5, 1.5)  # 顶点
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面边
        (0, 4), (1, 4), (2, 4), (3, 4)  # 侧边
    ]
    for i, pos in enumerate(vertices):
        G.add_node(i, pos=np.array(pos))
    G.add_edges_from(edges)
    return G


# 可视化函数 ---------------------------------------------------------
def plot_random_structure(graph, faces):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始结构
    positions = np.array([graph.nodes[n]['pos'] for n in graph.nodes])
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='black', s=50, depthshade=False)

    # 绘制边
    for u, v in graph.edges():
        line = np.array([graph.nodes[u]['pos'], graph.nodes[v]['pos']])
        ax.plot(line[:, 0], line[:, 1], line[:, 2],
                c='gray', alpha=0.3, linewidth=1)

    # 绘制找到的面
    if faces:
        colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))
        for i, face in enumerate(faces):
            polygon = [graph.nodes[n]['pos'] for n in face]
            poly = Poly3DCollection([polygon],
                                    facecolors=colors[i],
                                    alpha=0.6,
                                    edgecolor='k')
            ax.add_collection3d(poly)

    # 设置视角
    ax.view_init(elev=25, azim=-45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.title(f"Random 3D Structure: {len(faces)} Planar Faces Found")
    plt.tight_layout()
    plt.show()


# 主程序流程 ---------------------------------------------------------
if __name__ == "__main__":
    # 生成标准数据
    # cube_graph = create_cube_graph()
    # cube_graph = create_pyramid_graph()

    # 生成随机三维网络
    random_graph = generate_random_3d_graph(
        num_nodes=30,
        edge_prob=0.3,
        seed=42,  # 固定种子用于可重复性
        bbox_size=3.0
    )

    # 提取共面闭合环
    detected_faces = extract_polyhedral_blocks(random_graph)

    # 打印统计信息
    print(f"节点数: {random_graph.number_of_nodes()}")
    print(f"边数: {random_graph.number_of_edges()}")
    print(f"检测到共面闭合环数量: {len(detected_faces)}")

    # 可视化
    plot_random_structure(random_graph, detected_faces)

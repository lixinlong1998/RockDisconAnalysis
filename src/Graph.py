import numpy as np
import networkx as nx
from scipy.spatial import distance
from src import Segment
def find_intersection_point(seg1: Segment.Segment, seg2: Segment.Segment, tolerance=1e-6) -> Optional[np.ndarray]:
    """
    判断两个线段（seg1, seg2）是否相交，如果相交则返回交点，否则返回 None。
    这里我们使用线段-线段相交的基本几何方法（基于参数方程求解交点）。
    """
    p1, p2 = seg1.p1, seg1.p2
    p3, p4 = seg2.p1, seg2.p2

    # 线段方向
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(denom) < tolerance:  # 线段平行
        return None

    # 交点的参数 t1（参数方程）
    t1 = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    t2 = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom

    # 判断交点是否在两个线段范围内
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        intersection = p1 + t1 * d1
        return intersection
    return None


def build_graph(segments: List[Segment], tolerance=1e-6) -> nx.Graph:
    """
    根据交点构建图。节点为交点，边为相交的线段。
    """
    G = nx.Graph()

    # 用来存储已存在的交点
    points_map = {}

    def add_point(point):
        # 将交点加入节点
        point_tuple = tuple(np.round(point, decimals=6))  # 处理浮点数精度问题
        if point_tuple not in points_map:
            node_id = len(points_map)
            points_map[point_tuple] = node_id
            G.add_node(node_id, point=point)
        return points_map[point_tuple]

    # 构建边
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue
            intersection = find_intersection_point(seg1, seg2, tolerance)
            if intersection is not None:
                # 获取交点的节点编号
                node1 = add_point(seg1.p1)
                node2 = add_point(seg1.p2)
                node3 = add_point(seg2.p1)
                node4 = add_point(seg2.p2)

                # 将边添加到图中
                G.add_edge(node1, node3, segment=(seg1, seg2))
                G.add_edge(node1, node4, segment=(seg1, seg2))
                G.add_edge(node2, node3, segment=(seg1, seg2))
                G.add_edge(node2, node4, segment=(seg1, seg2))

    return G


def find_blocks(graph: nx.Graph) -> List[List[int]]:
    """
    查找潜在的blocks，blocks为图的连通分量。
    """
    return list(nx.connected_components(graph))


def visualize_graph(graph: nx.Graph):
    """
    可视化图，展示节点和边（基于 NetworkX 提供的可视化工具）。
    """
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(graph)  # 用 spring layout 布局节点
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color="skyblue")
    plt.show()


# ============================
# 使用示例
# ============================

# 假设 `segments` 已经由 get_segments() 方法获取
segments = get_segments(discontinuitys, extension=1.5, verbose=True)

# 构建图
G = build_graph(segments, tolerance=1e-6)

# 可视化图
visualize_graph(G)

# 查找并输出潜在的blocks（连通分量）
blocks = find_blocks(G)
print(f"Found {len(blocks)} potential blocks.")
for block in blocks:
    print(block)

import numpy as np
import pyvista as pv
import tetgen


def create_cube_with_pyramid_pit():
    """
    构造一个立方体，并在顶面中心凹入一个金字塔形状坑。
    返回:
        vertices: (N, 3) array
        faces: (M, 3) array
    """
    # 立方体8顶点 (单位立方体)
    v = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ])

    # 顶部中央点，向内凹陷（z < 1）
    v_pit = np.array([[0.5, 0.5, 0.8]])
    v_all = np.vstack([v, v_pit])
    v_p = 8  # index of pit

    # 面：其余5个面保持原状（用三角面片表示）
    # 底面
    bottom = [[0, 1, 2], [0, 2, 3]]
    # 侧面
    side1 = [[0, 1, 5], [0, 5, 4]]
    side2 = [[1, 2, 6], [1, 6, 5]]
    side3 = [[2, 3, 7], [2, 7, 6]]
    side4 = [[3, 0, 4], [3, 4, 7]]

    # 顶面修改为：四个三角形连接凹点
    top = [
        [4, 5, v_p],  # front
        [5, 6, v_p],  # right
        [6, 7, v_p],  # back
        [7, 4, v_p],  # left
    ]

    faces = bottom + side1 + side2 + side3 + side4 + top
    faces = np.array(faces)

    return v_all, faces


def tetrahedralize_surface(vertices, faces, show_plot=True):
    # 构建 PyVista PolyData
    flat_faces = np.hstack([[3, *tri] for tri in faces])
    surf = pv.PolyData(vertices, flat_faces)

    # 用 TetGen 剖分
    tgen = tetgen.TetGen(surf)
    tgen.tetrahedralize()
    tet_mesh = tgen.grid
    tet_points = tet_mesh.points

    # 提取四面体索引
    try:
        tet_elements = tet_mesh.cells_dict[pv.CellType.TETRA]
    except AttributeError:
        raise RuntimeError("未生成四面体单元，请确认输入 surface 是闭合多面体。")

    # 可视化
    if show_plot:
        p = pv.Plotter()
        p.add_mesh(tet_mesh, show_edges=True, show_scalar_bar=False, opacity=0.5)
        p.show()

        # 在主函数结尾添加此部分代码
        # 单独显示每个四面体
        colors = [
            'red', 'green', 'blue', 'cyan', 'magenta',
            'yellow', 'orange', 'purple', 'brown', 'gray'
        ]
        p = pv.Plotter()
        for i, tet in enumerate(tet_elements):
            tet_pts_local = tet_points[tet]
            # 每个四面体由 4 个顶点定义
            cells = np.hstack([[4, 0, 1, 2, 3]])
            local_grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], tet_pts_local)
            p.add_mesh(local_grid, show_edges=True, opacity=0.7, color=colors[i % len(colors)])
        p.show()

    return tet_points, tet_elements


if __name__ == "__main__":
    vertices, faces = create_cube_with_pyramid_pit()
    tet_pts, tet_elements = tetrahedralize_surface(vertices, faces, show_plot=True)
    print(tet_elements)

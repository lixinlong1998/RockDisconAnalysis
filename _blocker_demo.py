import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict, Tuple

import panel as pn
import trimesh
from scipy.spatial import ConvexHull
import pyvista as pv

pn.extension('vtk')

# ========== 数学与几何工具 ==========
TOL = 1e-6

def plane_triple_intersection(p1, p2, p3):
    # plane: n·x + d = 0
    n1, d1 = p1[:3], p1[3]
    n2, d2 = p2[:3], p2[3]
    n3, d3 = p3[:3], p3[3]
    A = np.vstack([n1, n2, n3])
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    b = -np.array([d1, d2, d3], dtype=float)
    x = np.linalg.solve(A, b)
    return x

def halfspace_feasible(point, planes, tol=TOL):
    vals = [np.dot(p[:3], point) + p[3] for p in planes]
    return all(v <= tol for v in vals)

def build_block_from_planes(planes: List[np.ndarray]):
    # 1) 三平面交点
    pts = []
    idx_triplets = []
    for (i,j,k) in combinations(range(len(planes)), 3):
        p = plane_triple_intersection(planes[i], planes[j], planes[k])
        if p is None:
            continue
        if halfspace_feasible(p, planes):
            pts.append(p)
            idx_triplets.append((i,j,k))
    if len(pts) < 4:
        return None, None, None

    pts = np.array(pts)
    # 去重（防止数值毛刺导致重复）
    uniq, inv = np.unique(np.round(pts, 8), axis=0, return_inverse=True)
    pts = uniq

    # 2) 用凸包生成三角面（凸块体时稳定）
    hull = ConvexHull(pts)
    faces = hull.simplices  # 每个元素是3顶点索引
    return pts, faces, hull

def export_ply(points, faces, out_path, color=(200,120,30)):
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    vc = np.tile(np.array(color, dtype=np.uint8), (len(points),1))
    mesh.visual.vertex_colors = np.hstack([vc, 255*np.ones((len(points),1), dtype=np.uint8)])
    mesh.export(out_path)
    return out_path

# ========== 数据结构 ==========
@dataclass
class PlaneItem:
    pid: str
    n: np.ndarray  # 法向 (3,)
    d: float       # 常数项
    center: np.ndarray  # 可视化中心
    size: float = 2.0   # 可视化平面大小（边长）

    def as_coeff(self) -> np.ndarray:
        return np.array([self.n[0], self.n[1], self.n[2], self.d], dtype=float)

# ========== Demo 场景（6 面盒子 + 两个斜切面做演示） ==========
def demo_planes_unit_box():
    # 盒子边界：-1 <= x,y,z <= 1
    # 平面写成 n·x + d <= 0 为“内侧”（选面时构造半空间）
    planes = []
    # x = 1  ->  n=(+1,0,0),  n·x + d <= 0  => x + d <= 0  内部是 x<=1 -> d = -1
    planes.append(PlaneItem('X+', np.array([+1,0,0]), -1.0, np.array([+1,0,0]), 2.2))
    # x = -1 ->  n=(-1,0,0), 内部是 x>=-1 -> (-1)*x + d <= 0 -> -x + d <=0 -> d= -1
    planes.append(PlaneItem('X-', np.array([-1,0,0]), -1.0, np.array([-1,0,0]), 2.2))
    # y = 1
    planes.append(PlaneItem('Y+', np.array([0,+1,0]), -1.0, np.array([0,+1,0]), 2.2))
    # y = -1
    planes.append(PlaneItem('Y-', np.array([0,-1,0]), -1.0, np.array([0,-1,0]), 2.2))
    # z = 1
    planes.append(PlaneItem('Z+', np.array([0,0,+1]), -1.0, np.array([0,0,+1]), 2.2))
    # z = -1
    planes.append(PlaneItem('Z-', np.array([0,0,-1]), -1.0, np.array([0,0,-1]), 2.2))

    # 额外两个演示面（可不选）：
    # 斜切面1：  x + y + z = 1.2  -> n=(1,1,1), d = -1.2
    n1 = np.array([1,1,1])/np.sqrt(3)
    d1 = -1.2/np.linalg.norm([1,1,1])  # 规范化后要匹配 d
    planes.append(PlaneItem('S1', n1, d1, np.array([0.4,0.4,0.4]), 3.0))
    # 斜切面2：  -x + 2y - z = 0.3
    n2 = np.array([-1,2,-1], dtype=float)
    n2 = n2 / np.linalg.norm(n2)
    d2 = -0.3 / np.linalg.norm([-1,2,-1])
    planes.append(PlaneItem('S2', n2, d2, np.array([-0.1,0.2,-0.1]), 3.0))

    return planes

# ========== PyVista + Panel 交互 ==========
class BlockPickerApp:
    def __init__(self):
        self.planes: List[PlaneItem] = demo_planes_unit_box()
        self.selected: Dict[str, bool] = {p.pid: False for p in self.planes}
        self.actor_map: Dict[str, str] = {}  # actor.mapper address -> pid
        self.block_mesh = None
        self.block_points = None
        self.block_faces = None

        # Widgets
        self.info = pn.pane.Markdown("**选择若干个相交的平面（≥3），然后点击“生成岩块”**", sizing_mode="stretch_width")
        self.btn_build = pn.widgets.Button(name="生成岩块", button_type="primary")
        self.btn_export = pn.widgets.Button(name="导出 PLY", button_type="success", disabled=True)
        self.out_text = pn.pane.Markdown("", sizing_mode="stretch_width")

        # 列出面
        self.chkboxes = pn.widgets.CheckBoxGroup(
            name="已选平面（也可直接点击 3D 中的平面切换选择）",
            options=[p.pid for p in self.planes],
            inline=False
        )

        # Plotter
        self.plotter = pv.Plotter(window_size=(900, 700))
        self._setup_scene()
        self.plot_pane = pn.pane.VTK(self.plotter.ren_win, sizing_mode="stretch_both")

        # Events
        self.chkboxes.param.watch(self._on_list_select, 'value')
        self.btn_build.on_click(self._on_build_block)
        self.btn_export.on_click(self._on_export)

    def _setup_scene(self):
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_bounds(grid='front', location='outer', all_edges=True)

        # 添加各平面
        for p in self.planes:
            mesh = pv.Plane(center=p.center, direction=p.n, i_size=p.size, j_size=p.size)
            actor = self.plotter.add_mesh(mesh, color="#bbbbbb", opacity=0.6, name=p.pid, pickable=True)
            # 用 actor 的唯一 id 关联到 pid（不同版本 PyVista 访问方式略有不同，这里用 id()）
            self.actor_map[id(actor)] = p.pid
            # 在平面中心放个文本
            self.plotter.add_point_labels([p.center], [p.pid], text_color='black', point_size=0, font_size=14)

        # 启用拾取：点击平面切换选中状态
        def _pick_callback(picked):
            # 返回的是 actor 列表或单个 actor
            if picked is None:
                return
            actors = picked if isinstance(picked, list) else [picked]
            for ac in actors:
                pid = self.actor_map.get(id(ac))
                if pid is None:
                    continue
                self.selected[pid] = not self.selected[pid]
                self._refresh_plane_color(pid)
            self._sync_list_from_selected()
            self.plotter.render()

        self.plotter.enable_mesh_picking(callback=_pick_callback, show=True, left_clicking=True, through=False)

        # 相机
        self.plotter.camera_position = 'iso'

    def _refresh_plane_color(self, pid: str):
        # 根据选中状态重绘颜色
        actor = self.plotter.renderer.find_actor(pid)
        if actor is None:
            return
        if self.selected.get(pid, False):
            actor.prop.color = (0.2, 0.6, 1.0)  # 选中：蓝
            actor.prop.opacity = 0.8
        else:
            actor.prop.color = (0.73, 0.73, 0.73)  # 未选：灰
            actor.prop.opacity = 0.6

    def _sync_list_from_selected(self):
        self.chkboxes.value = [pid for pid, v in self.selected.items() if v]

    def _on_list_select(self, event):
        # 从左侧复选框同步到 3D
        new_vals = set(event.new)
        for pid in self.selected.keys():
            self.selected[pid] = pid in new_vals
            self._refresh_plane_color(pid)
        self.plotter.render()

    def _on_build_block(self, _):
        sel_pids = [pid for pid, v in self.selected.items() if v]
        if len(sel_pids) < 3:
            self.out_text.object = "⚠️ 至少选择 3 个相交的平面。"
            return

        sel_planes = [next(p for p in self.planes if p.pid == pid) for pid in sel_pids]
        coeffs = [p.as_coeff() for p in sel_planes]

        pts, faces, hull = build_block_from_planes(coeffs)
        # 清理旧块体
        if self.block_mesh is not None:
            try:
                self.plotter.remove_actor("BLOCK_MESH")
            except Exception:
                pass
            self.block_mesh = None

        if pts is None:
            self.block_points = None
            self.block_faces = None
            self.btn_export.disabled = True
            self.out_text.object = "❌ 未形成闭合块体（可能是平面未围成有界区域，或法向/位置组合无解）。"
            return

        # 构造 PyVista PolyData（faces 需要 polyface 格式）
        faces_pv = np.hstack([np.array([3, *tri]) for tri in faces]).astype(np.int64)
        poly = pv.PolyData(pts, faces_pv)

        # 展示块体
        self.block_mesh = self.plotter.add_mesh(poly, name="BLOCK_MESH", color="#ffb000", opacity=0.7, show_edges=True)
        self.block_points = pts
        self.block_faces = faces
        self.btn_export.disabled = False

        # 计算几何属性
        tm = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
        vol = float(tm.volume)
        area = float(tm.area)
        self.out_text.object = f"✅ 生成成功！**体积**≈ `{vol:.4f}`，**表面积**≈ `{area:.4f}`。\n\n来源面：`{', '.join(sel_pids)}`"
        self.plotter.render()

    def _on_export(self, _):
        if self.block_points is None or self.block_faces is None:
            self.out_text.object = "⚠️ 没有可导出的块体。"
            return
        path = export_ply(self.block_points, self.block_faces, "block_demo_output.ply")
        self.out_text.object = f"💾 已导出：`{path}`（工作目录下）。"

    def view(self):
        sidebar = pn.Column(
            self.info,
            pn.pane.Markdown("### 平面选择"),
            self.chkboxes,
            pn.Spacer(height=10),
            pn.Row(self.btn_build, self.btn_export, sizing_mode="stretch_width"),
            pn.layout.Divider(),
            pn.pane.Markdown("### 输出"),
            self.out_text,
            width=360,
            sizing_mode="stretch_height"
        )
        main = pn.Row(self.plot_pane, sizing_mode="stretch_both")
        return pn.Row(sidebar, main, sizing_mode="stretch_both")

# ========== 启动 ==========
# ========== 启动 ==========
app = BlockPickerApp()

template = pn.template.FastListTemplate(
    title="交互式岩块定位 Demo（半空间法）",
    sidebar=[],
    main=[app.view()],
    theme="default",
)
template.servable()  # 供 `panel serve block_demo.py` 使用

# 若直接 `python block_demo.py` 运行，则启动内置server
if __name__ == "__main__":
    pn.serve(template, show=True)


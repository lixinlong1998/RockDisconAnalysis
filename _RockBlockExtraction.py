import os
import time
from src import Workflow
from src import DiconCharac
from src import Export

"""
📌 有数据及其数据结构
🗒️ TSDK_Rockfall_1_P2_0.05m_facets_converted.txt
X	Y	Z	cluster_id	subcluster_id	facet_id	A	B	C	D
🗒️ TSDK_Rockfall_1_P2_0.05m_facets_metrics.csv
facet_id,facet_rms,facet_points_number,A,B,C,D,Area,TraceLength,TraceStartX,TraceStartY,TraceStartZ,TraceEndX,TraceEndY,TraceEndZ
🗒️ TSDK_Rockfall_1_P2_0.05m_facets_stereographic_KDE_peaks.csv
cluster_id,theta,r,dip_deg,dipdir_deg,z
🗒️ TSDK_Rockfall_1_P2_0.05m_facets_stereographic_labels.csv
cluster_id,facet_id
📁 TSDK_Rockfall_1_P2_0.05m_facets_contours
每个facet polygon对应的ply文件,例如:
facet_id=6对应的文件 🗒️ facet_contour_006.ply, 其内容为:
    ply
    format ascii 1.0
    element vertex 6
    property float x
    property float y
    property float z
    element edge 6
    property int vertex1
    property int vertex2
    end_header
    33.22229766845703 2.4297146797180176 105.68933868408203
    33.25056076049805 2.3906333446502686 105.7930679321289
    33.30971145629883 2.3175976276397705 105.86898803710938
    33.39741897583008 2.2164437770843506 105.8661880493164
    33.419334411621094 2.1979339122772217 105.75639343261719
    33.427310943603516 2.1939399242401123 105.67214965820312
    0 1
    1 2
    2 3
    3 4
    4 5
    5 0

📌 加载结构面数据,进行分析

"""
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ################################################# input
    path_data = r'D:\Research\20250313_RockFractureSeg\Experiments\Exp2_DiscontinuityCharacterization\G3033_9_Part2_ClusterAnalysis xyz-js-c-abcd.txt'
    path_workspace = None
    project_name = 'G3033_9_Part2'

    # ################################################# output
    if not path_workspace:
        path_workspace = os.path.join(os.path.dirname(path_data), 'Workspace')
    os.makedirs(path_workspace, exist_ok=True)
    path_visualize = os.path.join(path_workspace, 'Visualize')
    os.makedirs(path_visualize, exist_ok=True)
    path_visualize_combinations = os.path.join(path_visualize, 'Combinations')
    os.makedirs(path_visualize_combinations, exist_ok=True)

    # ################################################# exportation
    path_group_discon_inliers = os.path.join(path_workspace, f'{project_name}_Inliers.ply')
    path_group_discon_polygons = os.path.join(path_workspace, f'{project_name}_Polygons.ply')
    path_group_discon_traces = os.path.join(path_workspace, f'{project_name}_Traces.ply')
    path_group_discon_ellipdisks = os.path.join(path_workspace, f'{project_name}_EllipDisks.ply')
    path_group_discon_normals = os.path.join(path_workspace, f'{project_name}_Normals.ply')
    path_group_discon_segments = os.path.join(path_workspace, f'{project_name}_Segments.ply')
    path_discon_analysis = os.path.join(path_workspace, f'{project_name}_Discontinuitys.pkl')
    path_discon_database = os.path.join(path_workspace, f'{project_name}_Discontinuitys.csv')

    # ################################################# main
    starttime = time.perf_counter()
    '''
    load DSE data
    '''
    data, total_pointcloud, clusters_pointcloud, cluster_id_dict = Workflow.load_pointcloud_DSE(path_data)

    '''
    rebuild data by cluster
    '''
    clusters = Workflow.get_clusters(data, total_pointcloud, clusters_pointcloud, cluster_id_dict)  # checked✅
    clusters = Workflow.ransac_planarity_filter(clusters)

    '''
    build discontinuity sets with basic information, such as dip and strike
    '''
    discontinuitys = Workflow.get_discontinuitys(clusters)  # checked✅
    print(f'[report] {len(discontinuitys.discontinuitys)} discontinuities have been found.')

    '''
    build discontinuity elliptical disk models
    '''
    # test single discontinuity_characterization
    for i in range(len(discontinuitys.discontinuitys)):
        # print(discontinuitys.discontinuitys[23].print_all_attributes())
        DiconCharac.discontinuity_characterization(discontinuitys.discontinuitys[i])  # 结构面椭圆模型的长轴之长等于迹线长度
        # if discontinuitys.discontinuitys[i].polygon_area == 0:
        #     print(discontinuitys.discontinuitys[i].print_all_attributes())
    # print(discontinuitys.discontinuitys[58].print_all_attributes())
    # discontinuitys = DiconCharac.multiprocess_discontinuity_characterization(discontinuitys, pool_size=1)
    # print(discontinuitys.discontinuitys[58].print_all_attributes())

    '''
    calculate segments
    '''
    # discontinuitys = Discontinuity.Discontinuitys()
    # discontinuitys.load_database(discontinuitys_databasepath)
    # segments = Segment.get_segments(discontinuitys, extention=1.5, pool_size=16, max_memory=32)  # 结构面椭圆模型扩展1.5倍

    '''
    build node-edge graph
    '''
    # vertex_graph, unodes_coord = BlockGen.get_vertex_graph(discontinuitys, segments, extention=1.5, pool_size=16,
    #                                                        max_memory=32)

    '''
    search rock blocks
    '''
    # rock_blocks = BlockGen.get_rockblocks(vertex_graph, unodes_coord)

    '''
    export for visualization and analysis
    '''
    # 结构面独立可视化
    Export.export_each_combination(path_visualize_combinations, discontinuitys)
    # 每类数据的分组可视化
    Export.export_all_inliers(path_group_discon_inliers, discontinuitys)
    Export.export_all_polygons(path_group_discon_polygons, discontinuitys)
    Export.export_all_traces(path_group_discon_traces, discontinuitys)
    Export.export_all_ellipdisks(path_group_discon_ellipdisks, discontinuitys)
    # Export.export_all_segments(path_group_discon_segments, segments)
    # 所有结构面的单个参数
    Export.export_discon_database(path_discon_analysis, discontinuitys, format='pkl')
    Export.export_discon_analysis(path_discon_database, discontinuitys, format='csv')

    print(f'[time cost]{Workflow.time_cost_hms(time.perf_counter() - starttime)} — All process have done.')

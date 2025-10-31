import time
import numpy as np
from src import PointCloud

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ################################################# input
    data_path = r'E:\Database\_RockPoints\TSDK_Rockfall_RegularClip\test\TSDK_Rockfall19_R1_0 xyz-js-c-abcd_separet.txt'


    # ################################################# main
    def time_cost_hms(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h} h {m} min {s:.2f} sec"


    starttime = time.perf_counter()

    # 读取数据（无表头，tab 分隔）
    data = np.loadtxt(data_path, delimiter='\t')

    # 提取所有 (js, cl) 并生成唯一 cluster_id
    js_list = [int(row[3]) for row in data]
    unique_joints = list(set(js_list))
    joints_pointcloud = {joint_id: PointCloud.RockPointCloud() for joint_id in unique_joints}

    js_cl_list = [(int(row[3]), int(row[4])) for row in data]
    unique_clusters = list(set(js_cl_list))
    cluster_id_dict = {key: idx for idx, key in enumerate(unique_clusters)}
    clusters_pointcloud = {key: PointCloud.RockPointCloud() for idx, key in enumerate(unique_clusters)}

    for point_id, row in enumerate(data):
        X, Y, Z = row[0:3]
        joint_id = int(row[3])
        joint_cluster_id = int(row[4])
        cluster_id = cluster_id_dict[(joint_id, joint_cluster_id)]
        R, G, B = 0, 0, 0
        a, b, c, d = row[5:9]
        joints_pointcloud[joint_id].add(X, Y, Z, point_id, joint_id, joint_cluster_id, cluster_id, R, G, B, a, b, c, d)

    # 提取感兴趣字段用于张量化处理
    print(f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — load point cloud to total_pointcloud.')

    for joint_id, pointcloud in joints_pointcloud.items():
        pointcloud.export(data_path.replace('.txt', f'_{int(joint_id)}.ply'), format='ply_bin')

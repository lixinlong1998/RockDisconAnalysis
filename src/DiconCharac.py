import math
import time
import multiprocessing
import numpy as np
from src import Discontinuity


def discontinuity_characterization(discontinuity: Discontinuity.Discontinuity):
    '''
    调用discontinuity类的方法以计算特征参数
    :param discontinuity:
    :return:
    '''

    # 预设定参数
    edge_method = 'convex'  # 'ashape' or 'convex'
    trace_method = 'edges'  # 'farthest2' or 'edges'
    disc_type = 'elliptical'  # 'elliptical' or 'circle'
    roughness_method = 'pca'  # 'zrange' or 'pca'

    start_time = time.perf_counter()
    # 寻找cluster点云的平面边界，注意如果只是为了寻找迹线，则直接计算凸包即可，因为凹点不可能成为迹线的端点
    # [每一步都需要检验discontinuity是否valid]
    if not discontinuity.valid:
        return discontinuity
    discontinuity.get_polygon(method=edge_method)  # checked

    # 提取迹线线段,默认迹线求解方法为farthest2
    # [每一步都需要检验discontinuity是否valid]
    if not discontinuity.valid:
        return discontinuity
    if trace_method == 'edges':
        discontinuity.get_trace_segment_from_edges()
    elif trace_method == 'farthest2':
        discontinuity.get_trace_segment_from_farthest2()  # checked

    # 计算结构面粗糙度
    if not discontinuity.valid:
        return discontinuity
    discontinuity.get_roughness(method=roughness_method)  # unchecked

    # 计算结构面圆盘模型
    if not discontinuity.valid:
        return discontinuity
    if disc_type == 'circle':
        discontinuity.get_disc_circle()
    elif disc_type == 'elliptical':
        discontinuity.get_disc_elliptical()

    # 计算开销
    discontinuity.calculate_time = time.perf_counter() - start_time

    return discontinuity


def print_error(value):
    '''
    这个函数可以输出多进程中的报错，但是不会终止多进程
    '''
    print("error: ", value)


def multiprocess_discontinuity_characterization(discontinuitys, pool_size):
    '''
    multiprocess_
    :param discontinuitys:
    :return:
    '''
    starttime = time.perf_counter()
    # creat multiprocessing pool with given process number
    pool = multiprocessing.Pool(processes=pool_size)

    # Divide discontinuitys into groups by pool_size.
    # 原始的discontinuity是按照点的数量进行从大到小排序的，此处需要根据discontinuity的points的数量进行均匀分配
    # 使用“轮流分配”的策略（round-robin），把排序后的 datalist 依次分发到不同的 batch 容器中
    datalist = discontinuitys.discontinuitys
    # 按点数量从大到小排序
    datalist_ranked = sorted(datalist, key=lambda d: len(d.rock_points.points), reverse=True)
    # 初始化 pool_size 个空 batch
    batches = [[] for _ in range(pool_size)]
    # 轮流将数据项依次分配到各 batch 中，确保每个 batch 的数据点数量尽量均衡
    for idx, item in enumerate(datalist_ranked):
        batches[idx % pool_size].append(item)
    batch_size = math.ceil(np.mean([len(batch) for batch in batches]))
    print(f'[report] pool_size:{pool_size}')
    print(f'[report] batch_size:{batch_size}')
    print(f'[report] batch_number:{len(batches)}')

    # For each batch, a process pool is started
    results = []
    for batch in batches:
        result = pool.starmap_async(discontinuity_characterization, [(discontinuity,) for discontinuity in batch],
                                    error_callback=print_error)
        results.append(result)

    # Wait for all process pools to finish executing
    print('[report] Waiting for all subprocesses done...')
    for result in results:
        result.wait()

    # 获取所有子进程返回的结果（二维列表），再拉平成一维
    true_results = [r.get() for r in results]  # => List of List
    flattened = [item for sublist in true_results for item in sublist]
    discontinuitys.discontinuitys = flattened  # 可选：更新结构体
    assert len(discontinuitys.discontinuitys) == len(flattened), "长度不一致，可能处理有误"

    # Closing the process pool
    pool.close()
    pool.join()
    print(
        f'[time cost]{time_cost_hms(time.perf_counter() - starttime)} — calculate characterization of discontinuities.')
    return discontinuitys


def time_cost_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h} h {m} min {s:.2f} sec"

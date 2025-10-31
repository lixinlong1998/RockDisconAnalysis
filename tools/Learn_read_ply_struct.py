import struct


def read_vertex(file_path, target_index):
    with open(file_path, 'rb') as f:
        # ----------------------------
        # 1. 解析头文件
        # ----------------------------
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line == 'end_header':
                header_end_pos = f.tell()  # 记录数据起始位置
                break

        # ----------------------------
        # 2. 计算顶点数据存储格式
        # ----------------------------
        # 根据属性定义计算每个顶点占用的字节数
        # 3个double(坐标) + 3个double(法线) + 3个uchar(颜色)
        vertex_format = '<3d3d3B'  # 小端字节序
        vertex_size = struct.calcsize(vertex_format)  # 自动计算字节数

        # ----------------------------
        # 3. 读取第target_index个顶点 (索引从0开始)
        # ----------------------------
        if target_index >= vertex_count:
            raise ValueError(f"顶点索引超出范围，文件只有{vertex_count}个顶点")

        # 定位到目标顶点起始位置
        f.seek(header_end_pos + target_index * vertex_size)

        # 读取并解包二进制数据
        data = f.read(vertex_size)
        x, y, z, nx, ny, nz, red, green, blue = struct.unpack(vertex_format, data)

        return {
            'position': (x, y, z),
            'normal': (nx, ny, nz),
            'color': (red, green, blue)
        }


# 使用示例
ply_file_path = r'D:\Research\20250313_RockFractureSeg\Experiments\Exp1_DiscontinuityExtraction\G3033_9_Part2 XYZ-HSV-early_classification.ply'
result = read_vertex(ply_file_path, 202)
print(f"坐标: {result['position']}")
print(f"法线: {result['normal']}")
print(f"颜色(RGB): {result['color']}")

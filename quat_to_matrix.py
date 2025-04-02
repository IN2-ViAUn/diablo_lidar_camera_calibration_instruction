import numpy as np
from scipy.spatial.transform import Rotation as R

def output(input_file, output_file):

    with open(input_file, 'r') as f:
        lines = f.readlines()

    print("lines:", len(lines))

    with open(output_file, 'w') as out:
        for line in lines:
            # 解析每一行的数据
            parts = list(map(float, line.strip().split(' ')))
            if len(parts) != 9:
                continue  # 忽略格式不正确的行

            _, _, qw, qx, qy, qz, x, y, z = parts

            # 创建旋转矩阵
            rotation = R.from_quat([qx, qy, qz, qw])  # scipy的顺序是 [x, y, z, w]
            rot_matrix = rotation.as_matrix()  # 3x3 旋转矩阵

            # 构建 4x4 变换矩阵
            rt_matrix = np.eye(4)
            rt_matrix[:3, :3] = rot_matrix
            rt_matrix[:3, 3] = [x, y, z]

            # 将矩阵按行主序展开写入文件
            flat_matrix = rt_matrix.flatten()
            out.write(' '.join(map(str, flat_matrix)) + '\n')

    print(f"转换完成，输出文件: {output_file}")

output("pose_quat.txt", "lidar_pose_matrix.txt")
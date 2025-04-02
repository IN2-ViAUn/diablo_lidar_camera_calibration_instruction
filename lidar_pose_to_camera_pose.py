import numpy as np

input_txt = "./lidar_pose_matrix.txt"
output_txt = "./camera_pose_matrix.txt"

matrix = np.array([
    [ 0.00181673,  0.34869249,  0.93723542, -0.08136078],
    [-0.99987711,  0.01522774, -0.00372722, -0.02777495],
    [-0.01557164, -0.93711347,  0.34867731,  0.14247277],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
# 计算逆矩阵
inv_matrix = np.linalg.inv(matrix)

# 假设 trajectories 是从 traj.txt 文件中读取的所有 4x4 矩阵组成的列表
with open(input_txt, 'r') as file:
    lines = file.readlines()

trajectories = []
for line in lines:
    # 将每行转换成 4x4 的矩阵
    matrix_line = np.fromstring(line, sep=' ').reshape(4, 4)
    trajectories.append(matrix_line)

# 应用逆矩阵并保存新轨迹到 trajnew.txt
with open(output_txt, 'w') as new_file:
    for traj in trajectories:
        transformed_traj =  traj @ matrix  # 左乘逆矩阵
        
        # 将变换后的矩阵展平成一维数组
        flat_transformed_traj = transformed_traj.flatten()
        
        # 将展平的数组转换为字符串，并去除多余的空格和换行符
        line_to_write = np.array2string(flat_transformed_traj, separator=' ', max_line_width=np.inf)[1:-1].replace('\n', '')
        new_file.write(line_to_write + '\n')
    new_file.flush()
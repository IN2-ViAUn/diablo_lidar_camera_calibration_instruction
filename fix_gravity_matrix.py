import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取4x4位姿矩阵
def read_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            if len(nums) == 16:
                pose = np.array(nums).reshape(4, 4)
                poses.append(pose)
    return poses

# 写入新的矩阵
def write_poses(poses, file_path):
    with open(file_path, 'w') as f:
        for pose in poses:
            flat = pose.reshape(-1)
            f.write(' '.join(f"{v:.8f}" for v in flat) + '\n')

# 对每个位姿应用局部四元数旋转
def rotate_poses_by_quaternion(poses, quat_local_rot):
    rotated_poses = []
    r_local = R.from_quat(quat_local_rot)

    for pose in poses:
        R_orig = pose[:3, :3]
        t = pose[:3, 3]

        r_orig = R.from_matrix(R_orig)
        r_new = r_orig * r_local  # 局部旋转，右乘

        pose_new = np.eye(4)
        pose_new[:3, :3] = r_new.as_matrix()
        pose_new[:3, 3] = t
        rotated_poses.append(pose_new)

    return rotated_poses

# 主程序
if __name__ == "__main__":
    input_file = "camera_pose_matrix.txt"
    output_file = "camera_pose_matrix_fixed.txt"

    # =======================
    # 原始四元数（wxyz），绕Y轴约20°
    qw, qx, qy, qz = 0.98477, -0.000886997, 0.173858, 0.000156597
    q_orig = R.from_quat([qx, qy, qz, qw])

    # 1️⃣ 坐标系变换：将Y轴 → X轴，相当于绕 Z 轴 -90°
    q_T = R.from_euler('z', 90, degrees=True)

    # 2️⃣ 变换四元数作用的坐标系：q' = q_T * q * q_T⁻¹
    q_transformed = q_T * q_orig * q_T.inv()

    # 3️⃣ 得到新的四元数 [x, y, z, w] 格式供Scipy使用
    quat_local_rot = q_transformed.as_quat()

    # ✅ 应用于轨迹每一帧
    poses = read_poses(input_file)
    rotated_poses = rotate_poses_by_quaternion(poses, quat_local_rot)
    write_poses(rotated_poses, output_file)

    print("原始四元数绕 Y → 转为绕 X")
    print("转换后的四元数（wxyz）:", [quat_local_rot[3], quat_local_rot[0], quat_local_rot[1], quat_local_rot[2]])
    print(f"已保存旋转后的位姿到: {output_file}")

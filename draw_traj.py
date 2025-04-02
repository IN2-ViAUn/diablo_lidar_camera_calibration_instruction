import numpy as np
import open3d as o3d

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

# 创建轨迹线段（只表示路径，不包含朝向）
def create_trajectory_line_set(poses):
    points = [pose[:3, 3] for pose in poses]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

# 创建每个位姿的坐标系（用RGB轴表示朝向）
def create_coordinate_frames(poses, size=0.05):
    frames = []
    for pose in poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(pose)
        frames.append(frame)
    return frames

# 主程序
if __name__ == "__main__":
    file_path = "camera_pose_matrix.txt"
    poses = read_poses(file_path)

    trajectory = create_trajectory_line_set(poses)
    frames = create_coordinate_frames(poses, size=0.05)

    o3d.visualization.draw_geometries([trajectory, *frames],
                                      window_name="Trajectory with Orientation (RGB Axes)",
                                      point_show_normal=False)

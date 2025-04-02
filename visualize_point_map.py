import numpy as np
import open3d as o3d
import cv2
import os

def load_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = list(map(float, line.split()))
            if len(numbers) == 16:
                pose = np.array(numbers).reshape(4, 4)
                poses.append(pose)
    return poses

def create_camera_geometry(scale=0.1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0, 0, 0])

def visualize_point_cloud_interactively(point_cloud_data):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis_objects = []
    trajectory = []

    for i, (points, colors, pose) in enumerate(point_cloud_data):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis_objects.append(pcd)

        camera_mesh = create_camera_mesh(pose)
        vis_objects.append(camera_mesh)

        camera_geom = create_camera_geometry(scale=0.1)
        camera_geom.transform(pose)
        vis.add_geometry(camera_geom)

        trajectory.append(pose[:3, 3])
        if len(trajectory) > 1:
            trajectory_line = o3d.geometry.LineSet()
            trajectory_points = np.array(trajectory)
            trajectory_lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
            trajectory_line.points = o3d.utility.Vector3dVector(trajectory_points)
            trajectory_line.lines = o3d.utility.Vector2iVector(trajectory_lines)
            vis_objects.append(trajectory_line)

        vis.add_geometry(pcd)
        vis.add_geometry(camera_mesh)

        vis.poll_events()
        vis.update_renderer()
        print(f"Frame {i + 1}/{len(point_cloud_data)} added. Press 'N' for next frame or 'Q' to quit.")

        while True:
            key = input("Enter 'N' for next frame or 'Q' to quit: ").strip().lower()
            if key == 'n':
                break
            elif key == 'q':
                vis.destroy_window()
                return

    vis.destroy_window()

def create_point_cloud_from_depth(depth_image, color_image, intrinsics, pose, downsample_factor=20, depth_scale=1000.0):
    fx, fy, cx, cy = intrinsics
    depth_image = depth_image.astype(np.float32) / depth_scale
    h, w = depth_image.shape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth_image.flatten()
    valid = (z_flat > 0)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    z_flat = z_flat[valid]
    x_3d = (x_flat - cx) * z_flat / fx
    y_3d = (y_flat - cy) * z_flat / fy
    z_3d = z_flat
    points = np.vstack((x_3d, y_3d, z_3d)).T
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = (pose @ points_homogeneous.T).T
    points_transformed = points_transformed[:, :3]
    downsampled_points = points_transformed[::downsample_factor]
    color_image = color_image.astype(np.float32) / 255.0
    colors = color_image.reshape(-1, 3)[valid]
    downsampled_colors = colors[::downsample_factor]
    return downsampled_points, downsampled_colors

def create_camera_mesh(pose, scale=0.03):
    camera = o3d.geometry.LineSet()
    points = np.array([
        [0, 0, 0],
        [scale, scale, scale],
        [scale, -scale, scale],
        [-scale, -scale, scale],
        [-scale, scale, scale]
    ])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    camera.points = o3d.utility.Vector3dVector((pose[:3, :3] @ points.T).T + pose[:3, 3])
    camera.lines = o3d.utility.Vector2iVector(lines)
    return camera

def create_point_cloud_from_depth_folder(depth_folder, color_folder, pose_file, intrinsics, diffs, downsample_factor=10, number=100, depth_scale=1000.0, interval=1):
    poses = load_poses(pose_file)
    point_cloud_data = []
    depth_files = sorted(os.listdir(depth_folder), key=lambda x: int(x.split('.')[0]))
    color_files = sorted(os.listdir(color_folder), key=lambda x: int(x.split('.')[0]))
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])

    for i, depth_file in enumerate(depth_files[:number]):
        i = i * interval
        if i >= len(poses):
            break
        depth_image = cv2.imread(os.path.join(depth_folder, depth_file), cv2.IMREAD_UNCHANGED)
        color_image = cv2.imread(os.path.join(color_folder, color_files[i]), cv2.IMREAD_COLOR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if color_image.shape[0] != depth_image.shape[0] or color_image.shape[1] != depth_image.shape[1]:
            color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))
        pose = poses[i]
        point_cloud, colors = create_point_cloud_from_depth(depth_image, color_image, intrinsics, pose, downsample_factor, depth_scale)
        point_cloud_data.append((point_cloud, colors, pose))

    return point_cloud_data

class InteractivePointCloudViewer:
    def __init__(self, point_cloud_data, width=1280, height=720, view_scale=1.0, intrisics=None):
        self.point_cloud_data = point_cloud_data
        self.current_frame = -1
        self.view_scale = view_scale
        self.width = width
        self.height = height
        self.xyz_per_trans = 0.005
        self.object_view_mode = False
        self.trajectory = []
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=int(width * view_scale), height=int(height * view_scale), visible=True)
        self.vis.register_key_callback(ord("N"), self.on_N_key_press)
        self.vis.register_key_callback(ord("Q"), self.on_Q_key_press)
        self.vis.register_key_callback(ord("T"), self.on_T_key_press)
        self.pcd_combined = o3d.geometry.PointCloud()
        self.trajectory_line = o3d.geometry.LineSet()
        self.camera_meshs = []
        self.view_control = self.vis.get_view_control()
        fx, fy, cx, cy = intrisics
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        cparams = o3d.camera.PinholeCameraParameters()
        cparams.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        self.view_control.convert_from_pinhole_camera_parameters(cparams)
        self.vis.update_renderer()

    def on_N_key_press(self, vis):
        cam_param = self.view_control.convert_to_pinhole_camera_parameters()
        ext = cam_param.extrinsic
        self.current_frame += 1
        if self.current_frame >= len(self.point_cloud_data):
            return
        points, colors, pose = self.point_cloud_data[self.current_frame]
        points = points[::10]
        colors = colors[::10]
        self.pcd_combined.points.extend(o3d.utility.Vector3dVector(points))
        self.pcd_combined.colors.extend(o3d.utility.Vector3dVector(colors))
        self.trajectory.append(pose[:3, 3])
        if len(self.trajectory) > 1:
            trajectory_points = np.array(self.trajectory)
            trajectory_lines = [[i, i + 1] for i in range(len(self.trajectory) - 1)]
            self.trajectory_line.points = o3d.utility.Vector3dVector(trajectory_points)
            self.trajectory_line.lines = o3d.utility.Vector2iVector(trajectory_lines)
        camera_mesh = create_camera_mesh(pose)
        self.camera_meshs.append(camera_mesh)
        camera_geom = create_camera_geometry(scale=0.1)
        camera_geom.transform(pose)
        self.camera_meshs.append(camera_geom)
        vis.clear_geometries()
        vis.add_geometry(self.pcd_combined)
        vis.add_geometry(self.trajectory_line)
        for camera_mesh in self.camera_meshs:
            vis.add_geometry(camera_mesh)
        cam_param = self.view_control.convert_to_pinhole_camera_parameters()
        cam_param.extrinsic = ext
        self.view_control.convert_from_pinhole_camera_parameters(cam_param)
        vis.update_renderer()

    def on_Q_key_press(self, vis):
        vis.destroy_window()

    def on_T_key_press(self, vis):
        pass

    def run(self):
        self.vis.run()

if __name__ == "__main__":
    distCoeffs = np.array([0.458331, -2.87024, 0.00048617, -9.82262e-05, 1.74446, 0.331466, -2.66947, 1.65473], dtype=np.float32)
    depth_folder = "D:\\Coding\\rosbag_save_3_29_scannet\\diablo_rosbag_save\\depth_images"
    color_folder = "D:\\Coding\\rosbag_save_3_29_scannet\\diablo_rosbag_save\\rgb_images"
    pose_file = "./camera_pose_matrix_fixed.txt"
    intrinsics = [607.88, 607.966, 641.875, 365.585]
    depth_scale = 1000.0
    number = -1
    interval = 1
    point_cloud_data = create_point_cloud_from_depth_folder(depth_folder, color_folder, pose_file, intrinsics, diffs=distCoeffs, number=number, depth_scale=depth_scale, interval=interval)
    viewer = InteractivePointCloudViewer(point_cloud_data, intrisics=intrinsics)
    viewer.run()

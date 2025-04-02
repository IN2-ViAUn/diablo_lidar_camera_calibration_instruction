import cv2
import numpy as np
import os

# 相机内参矩阵 K
K = np.array([
    [607.879638671875, 0.0, 641.8748168945312],
    [0.0, 607.9664916992188, 365.58526611328125],
    [0.0, 0.0, 1.0]
])

# 畸变系数 D
# distortion_model 是 rational_polynomial，对应 OpenCV 支持的 model
# D 有 8 个参数：k1, k2, p1, p2, k3, k4, k5, k6
D = np.array([
    0.45833128690719604,
    -2.8702433109283447,
    0.00048616970889270306,
    -9.822617721511051e-05,
    1.744455099105835,
    0.3314657211303711,
    -2.6694741249084473,
    1.654725432395935
])

def undistort_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 假设图像大小为 1280x720（来自 camera_info）
    image_size = (1280, 720)

    # 计算优化后的新相机矩阵（optional: 保留更多边缘）
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, image_size, alpha=0)

    # 预计算 map
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, image_size, cv2.CV_16SC2)

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 去畸变
            undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

            # 保存
            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, undistorted)

    print("去畸变完成！")

# 用法示例
undistort_images("./png", "./png_undistortion")
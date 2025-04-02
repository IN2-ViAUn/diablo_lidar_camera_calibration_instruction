#include <iostream>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <输入文件夹> <输出文件夹> <intensity范围，例如: 10.0,50.0>" << std::endl;
        return -1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    float min_intensity, max_intensity;
    if (sscanf(argv[3], "%f,%f", &min_intensity, &max_intensity) != 2) {
        std::cerr << "无法解析 intensity 范围，格式应为: min,max" << std::endl;
        return -1;
    }

    // 创建输出目录（如果不存在）
    fs::create_directories(output_folder);

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".pcd") {
            std::string filename = entry.path().filename().string();
            std::cout << "处理文件: " << filename << std::endl;

            // 加载点云
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            if (pcl::io::loadPCDFile<pcl::PointXYZI>(entry.path().string(), *cloud) == -1) {
                std::cerr << "无法读取文件: " << filename << std::endl;
                continue;
            }

            // 过滤 intensity 范围
            pcl::PassThrough<pcl::PointXYZI> pass;
            pass.setInputCloud(cloud);
            pass.setFilterFieldName("intensity");
            pass.setFilterLimits(min_intensity, max_intensity);
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
            pass.filter(*cloud_filtered);

            // 输出文件路径
            std::string output_path = output_folder + "/" + filename;

            // 保存过滤后的点云
            pcl::io::savePCDFileBinary(output_path, *cloud_filtered);
            std::cout << "已保存: " << output_path << " (" << cloud_filtered->points.size() << " 点)" << std::endl;
        }
    }

    std::cout << "处理完成。" << std::endl;
    return 0;
}
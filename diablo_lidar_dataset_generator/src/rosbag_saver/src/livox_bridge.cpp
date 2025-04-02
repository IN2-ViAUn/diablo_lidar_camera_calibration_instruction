#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "livox_ros_driver2/msg/custom_msg.hpp"

class LivoxToPointCloud2Node : public rclcpp::Node
{
public:
    LivoxToPointCloud2Node()
        : Node("livox_to_pointcloud2_node")
    {
        sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            "/lidar/scan", 10,
            std::bind(&LivoxToPointCloud2Node::customMsgCallback, this, std::placeholders::_1));

        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/lidar", 10);
    }

private:
    void customMsgCallback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header = msg->header;
        cloud_msg.height = 1;
        cloud_msg.width = msg->point_num;

        cloud_msg.fields.resize(3);

        cloud_msg.fields[0].name = "x";
        cloud_msg.fields[0].offset = 0;
        cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[0].count = 1;

        cloud_msg.fields[1].name = "y";
        cloud_msg.fields[1].offset = 4;
        cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[1].count = 1;

        cloud_msg.fields[2].name = "z";
        cloud_msg.fields[2].offset = 8;
        cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        cloud_msg.fields[2].count = 1;

        cloud_msg.is_bigendian = false;
        cloud_msg.point_step = 12;  // 3 x float32 = 12 bytes per point
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
        cloud_msg.is_dense = true;
        cloud_msg.data.resize(cloud_msg.row_step);

        // 填充点云数据
        for (size_t i = 0; i < msg->point_num; ++i)
        {
            const auto& pt = msg->points[i];
            float x = pt.x;
            float y = pt.y;
            float z = pt.z;

            uint8_t* ptr = &cloud_msg.data[i * cloud_msg.point_step];
            memcpy(ptr + 0, &x, sizeof(float));
            memcpy(ptr + 4, &y, sizeof(float));
            memcpy(ptr + 8, &z, sizeof(float));
        }

        pub_->publish(cloud_msg);
    }

    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxToPointCloud2Node>());
    rclcpp::shutdown();
    return 0;
}

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <filesystem>

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;

class SyncSaverNode : public rclcpp::Node
{
public:
  SyncSaverNode()
      : Node("sync_saver_node"), count_(0)
  {
    std::filesystem::create_directories("output");

    image_sub_.subscribe(this, "/rgb/image_raw");
    cloud_sub_.subscribe(this, "/livox/lidar");
    depth_sub_.subscribe(this, "/depth_to_rgb/image_raw");

    sync_ = std::make_shared<Sync>(Sync(10), image_sub_, cloud_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&SyncSaverNode::callback, this, _1, _2, _3));

    RCLCPP_INFO(this->get_logger(), "SyncSaverNode with depth started");
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image,
      sensor_msgs::msg::PointCloud2,
      sensor_msgs::msg::Image>
      SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<Sync> sync_;

  int count_;
  pcl::PointCloud<pcl::PointXYZI> accumulated_cloud_;

  void callback(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg,
                const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg)
  {
    count_++;
    const int mod = count_ % 5;

    if (mod != 3 && mod != 4 && mod != 0) {
      return;
    }

    // Convert point cloud
    pcl::PointCloud<pcl::PointXYZI> current_cloud;
    pcl::fromROSMsg(*cloud_msg, current_cloud);
    accumulated_cloud_ += current_cloud;

    if (mod == 0) {
      // Save RGB image
      try {
        cv::Mat image = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
        std::stringstream ss_img;
        ss_img << "output/" << count_ << ".png";
        cv::imwrite(ss_img.str(), image);
      } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception for RGB: %s", e.what());
        return;
      }

      // Save Depth image
      try {
        cv::Mat depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
        std::stringstream ss_depth;
        ss_depth << "output/" << count_ << "_depth.png";
        cv::imwrite(ss_depth.str(), depth);
      } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception for depth: %s", e.what());
        return;
      }

      // Save accumulated point cloud
      std::stringstream ss_pcd;
      ss_pcd << "output/" << count_ << ".pcd";
      pcl::io::savePCDFileBinary(ss_pcd.str(), accumulated_cloud_);

      RCLCPP_INFO(this->get_logger(), "Saved RGB: %d.png, Depth: %d_depth.png, PCD: %d.pcd", count_, count_, count_);

      accumulated_cloud_.clear();
    }
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SyncSaverNode>());
  rclcpp::shutdown();
  return 0;
}

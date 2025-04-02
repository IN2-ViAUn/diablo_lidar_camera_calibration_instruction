#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class PoseImageSyncNode : public rclcpp::Node
{
public:
  PoseImageSyncNode() : Node("pose_image_sync_node"), gravity_pose_saved_(false)
  {
    // Define and create save directories
    save_root_path_ = "/home/bobh/diablo_rosbag_save";
    pose_quat_path_ = save_root_path_ / "pose_quat.txt";
    rgb_image_path_ = save_root_path_ / "rgb_images";
    depth_image_path_ = save_root_path_ / "depth_images";
    init_gravity_pose_path_ = save_root_path_ / "init_gravity_pose.txt";

    fs::create_directories(rgb_image_path_);
    fs::create_directories(depth_image_path_);

    // Subscribers for synchronized messages
    pose_sub_.subscribe(this, "/state_estimation");
    depth_sub_.subscribe(this, "/depth_to_rgb/image_raw");
    rgb_sub_.subscribe(this, "/rgb/image_raw");

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        nav_msgs::msg::Odometry,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image>;

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(300), pose_sub_, depth_sub_, rgb_sub_);
    sync_->registerCallback(std::bind(&PoseImageSyncNode::callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    // Subscriber for init gravity pose
    gravity_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/init_gravity_pose", 10,
        std::bind(&PoseImageSyncNode::gravity_pose_callback, this, std::placeholders::_1));
  }

  ~PoseImageSyncNode()
  {
    SaveJSON();
  }

private:
  void callback(const nav_msgs::msg::Odometry::ConstSharedPtr &pose,
                const sensor_msgs::msg::Image::ConstSharedPtr &depth_img,
                const sensor_msgs::msg::Image::ConstSharedPtr &rgb_img)
  {
    static int index = 0;
    RCLCPP_INFO(this->get_logger(), "Received synchronized data");

    save_pose_quaternion(pose);
    save_image(depth_img, depth_image_path_, index);
    save_image(rgb_img, rgb_image_path_, index);

    index++;
  }

  void gravity_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (gravity_pose_saved_)
      return;

    std::ofstream file(init_gravity_pose_path_);
    if (!file) {
      RCLCPP_ERROR(this->get_logger(), "Unable to open init_gravity_pose.txt for writing");
      return;
    }

    file << msg->pose.position.x << " "
         << msg->pose.position.y << " "
         << msg->pose.position.z << " "
         << msg->pose.orientation.w << " "
         << msg->pose.orientation.x << " "
         << msg->pose.orientation.y << " "
         << msg->pose.orientation.z << "\n";

    gravity_pose_saved_ = true;
    RCLCPP_INFO(this->get_logger(), "Saved initial gravity pose.");
  }

  void save_pose_quaternion(const nav_msgs::msg::Odometry::ConstSharedPtr &pose)
  {
    std::ofstream file(pose_quat_path_, std::ios_base::app);
    if (!file) {
      RCLCPP_ERROR(this->get_logger(), "Unable to open quaternion file for writing");
      return;
    }

    file << pose->header.stamp.sec << " "
         << pose->header.stamp.nanosec << " "
         << pose->pose.pose.orientation.w << " "
         << pose->pose.pose.orientation.x << " "
         << pose->pose.pose.orientation.y << " "
         << pose->pose.pose.orientation.z << " "
         << pose->pose.pose.position.x << " "
         << pose->pose.pose.position.y << " "
         << pose->pose.pose.position.z << "\n";
  }

  void save_image(const sensor_msgs::msg::Image::ConstSharedPtr msg, const fs::path &folder, int index)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      if (msg->encoding == "16UC1")
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
      }
      else if (msg->encoding == "32FC1")
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        cv::Mat depth_mm;
        cv_ptr->image.convertTo(depth_mm, CV_16UC1, 1000.0);
        cv::threshold(depth_mm, depth_mm, 65535, 65535, cv::THRESH_TRUNC);
        cv_ptr->image = depth_mm;
      }
      else
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      }
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    fs::path filename = folder / (std::to_string(index) + ".png");
    cv::imwrite(filename.string(), cv_ptr->image);

    json_array_.push_back({
        {"timestamp_sec", msg->header.stamp.sec},
        {"timestamp_nsec", msg->header.stamp.nanosec},
        {"filename", filename.string()}
    });
  }

  void SaveJSON()
  {
    fs::path json_path = save_root_path_ / "timestamps.json";
    std::ofstream file(json_path);
    if (file)
    {
      file << json_array_.dump(4);
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to open %s for writing", json_path.string().c_str());
    }
  }

  // Members
  message_filters::Subscriber<nav_msgs::msg::Odometry> pose_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
    nav_msgs::msg::Odometry, sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gravity_pose_sub_;
  bool gravity_pose_saved_;
  fs::path save_root_path_;
  fs::path pose_quat_path_;
  fs::path rgb_image_path_;
  fs::path depth_image_path_;
  fs::path init_gravity_pose_path_;
  nlohmann::json json_array_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PoseImageSyncNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

cmake_minimum_required(VERSION 3.8)
project(rosbag_image_save)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)  # 添加 nav_msgs 依赖
find_package(jsoncpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(message_filters REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

add_executable(${PROJECT_NAME}_node src/my_rosbag.cpp)

add_executable(livox_bridge_node src/livox_bridge.cpp)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  sensor_msgs
  cv_bridge
  geometry_msgs
  jsoncpp
  OpenCV 
  message_filters
  nav_msgs
  livox_ros_driver2
)

ament_target_dependencies(livox_bridge_node
  rclcpp
  sensor_msgs
  cv_bridge
  geometry_msgs
  jsoncpp
  OpenCV 
  message_filters
  nav_msgs
  livox_ros_driver2
)


install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS livox_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
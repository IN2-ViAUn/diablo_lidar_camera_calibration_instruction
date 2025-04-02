cmake_minimum_required(VERSION 3.10)
project(IntensityFilter)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(intensity_filter main.cpp)
target_link_libraries(intensity_filter ${PCL_LIBRARIES})

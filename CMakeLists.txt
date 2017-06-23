cmake_minimum_required(VERSION 2.8)
project(MySlam)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable (MySlam src/RunSlam.cpp)

target_link_libraries(MySlam ${OpenCV_LIBS})
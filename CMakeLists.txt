# CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project(desk_detector)

find_package(OpenCV 4.8 REQUIRED)

add_executable(desk_detector main.cpp)
target_link_libraries(desk_detector ${OpenCV_LIBS})

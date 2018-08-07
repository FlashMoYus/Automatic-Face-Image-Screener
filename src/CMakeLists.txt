cmake_minimum_required(VERSION 3.10)
project(ProperImageScreener)

set(CMAKE_CXX_STANDARD 17)

add_executable(ProperImageScreener main.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(ProperImageScreener ${OpenCV_LIBS})
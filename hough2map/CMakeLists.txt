cmake_minimum_required(VERSION 2.8.3)
project(hough2map)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

cs_add_executable(${PROJECT_NAME}
  src/node.cpp
)

cs_add_library(${PROJECT_NAME}_lib
  src/detector.cpp
)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

cs_install()
cs_export()

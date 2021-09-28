cmake_minimum_required(VERSION 2.8.3)
project(hough2map)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_package(OpenMP)


cs_add_executable(${PROJECT_NAME}
  src/detector.cpp
  src/node.cpp
)

cs_add_library(${PROJECT_NAME}_lib
  src/detector.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hough2map PUBLIC OpenMP::OpenMP_CXX)
endif()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib -fopenmp)

cs_install()
cs_export()

cmake_minimum_required(VERSION 3.5)
project(evaluation_deeplocaldesc)

find_package(OpenMP)
find_package(OpenCV REQUIRED)

set(CMAKE_C_FLAGS "-Wall ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++0x ${OpenMP_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})
# You have to specify -DOPENBLAS_INCLUDE_DIR=</path/to/OpenBLAS/include>
include_directories(${OPENBLAS_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

set(SOURCE_FILES main.cpp)
add_executable(main ${SOURCE_FILES})
target_link_libraries(main opencv_core opencv_imgproc opencv_imgcodecs opencv_xfeatures2d opencv_features2d openblas gfortran pthread)
cmake_minimum_required(VERSION 3.1)

project(bev2seg_2_tracking
    VERSION 1.0
    DESCRIPTION "Tracking utility for BEV2Seg_2"
    LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# Find Dependencies
find_package(OpenCV REQUIRED)
find_package(GTL REQUIRED)

# Add Target
add_executable(opencv_test src/opencv_test.cpp)
target_compile_options(opencv_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bev2seg_2_tracking src/bev2seg_2_tracking.cpp)
target_compile_options(bev2seg_2_tracking PRIVATE -Wall -Wextra -Wpedantic)


# Linking dependencies
target_include_directories(opencv_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_test PRIVATE ${OpenCV_LIBS})

target_include_directories(bev2seg_2_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(bev2seg_2_tracking PRIVATE ${GTL_INCLUDE_DIRS})
target_link_libraries(bev2seg_2_tracking gtl_core ${OpenCV_LIBS})


cmake_minimum_required(VERSION 3.16)
project(TreeLocPP LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_library(treelocpp
    src/config.cpp
    src/descriptors.cpp
    src/eval.cpp
    src/geometry.cpp
    src/io.cpp
    src/matching.cpp
)

target_include_directories(treelocpp PUBLIC include)
target_link_libraries(treelocpp PUBLIC Eigen3::Eigen)

add_executable(treelocpp_intra apps/intra.cpp)
target_link_libraries(treelocpp_intra PRIVATE treelocpp)

add_executable(treelocpp_inter apps/inter.cpp)
target_link_libraries(treelocpp_inter PRIVATE treelocpp)

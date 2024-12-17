cmake_minimum_required(VERSION 3.9)
project(main)
 
set(CMAKE_CXX_STANDARD 11)
 
#find_library(OpenCV)
find_package(OpenCV)
 
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR}) 
#add_executable(easycnn core.cc ylog.cc)
add_executable(easycnn train.cc)
target_link_libraries(easycnn ${OpenCV_LIBS})

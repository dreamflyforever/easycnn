cmake_minimum_required(VERSION 3.9)
project(main)
 
set(CMAKE_CXX_STANDARD 11)
 
# Add system include paths for macOS
include_directories(
    ${CMAKE_SOURCE_DIR}
    /usr/local/include
    /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
    /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
 
#find_library(OpenCV)
find_package(OpenCV)
 
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(easycnn train.cc)
target_link_libraries(easycnn ${OpenCV_LIBS})

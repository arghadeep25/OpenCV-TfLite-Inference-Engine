cmake_minimum_required(VERSION 3.16)
project(opencv_tflite_inference VERSION 0.1 DESCRIPTION "Security Camera" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DPROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")

# Set build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc features2d REQUIRED)


set(GLOG_INCLUDE_DIR "/usr/include")
set(GLOG_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu")

set(GFLAGS_INCLUDE_DIR "/usr/include")
set(GFLAGS_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu")


set(TENSORFLOW_SOURCE_DIR "${CMAKE_SOURCE_DIR}/submodules/tensorflow")
set(GOOGLE_TEST_SOURCE_DIR "${CMAKE_SOURCE_DIR}/submodules/googletest")
message(STATUS "Tensorflow source dir: ${TENSORFLOW_SOURCE_DIR}")
message(STATUS "Google Test source dir: ${GOOGLE_TEST_SOURCE_DIR}")

# Check if the directory exists
if (NOT EXISTS "${TENSORFLOW_SOURCE_DIR}/tensorflow")
    message(STATUS "TensorFlow directory does not exist. Try git submodule update --init --recursive")
else ()
    message(STATUS "TensorFlow directory found.")
endif ()

if (NOT EXISTS "${GOOGLE_TEST_SOURCE_DIR}/googletest")
    message(STATUS "Google Test directory does not exist. Try git submodule update --init --recursive")
else ()
    message(STATUS "Google Test directory found.")
endif ()

add_subdirectory(
        "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# Include OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIR})

# Source and include directories
SET(SRC_DIR src)
SET(INCLUDE_DIR tflite_inference_engine)

# Glob for source and header files
FILE(GLOB SRC_FILES ${SRC_DIR}/*.cpp)
FILE(GLOB INCLUDE_FILES ${INCLUDE_DIR}/*.hpp)

# Create a library from your source files
add_library(tflite_inference_engine_lib ${SRC_FILES} ${INCLUDE_FILES})
target_include_directories(tflite_inference_engine_lib PUBLIC ${INCLUDE_DIR})
target_link_libraries(tflite_inference_engine_lib ${OpenCV_LIBRARIES} tensorflow-lite ${GLOG_LIBRARY_DIR}/libglog.so ${GFLAGS_LIBRARY_DIR}/libgflags.so)

# Create the main executable
add_executable(tflite_inference_engine ${SRC_FILES} ${INCLUDE_FILES})
target_link_libraries(tflite_inference_engine tflite_inference_engine_lib tensorflow-lite ${OpenCV_LIBRARIES} ${GLOG_LIBRARY_DIR}/libglog.so ${GFLAGS_LIBRARY_DIR}/libgflags.so)

add_subdirectory(tests)
add_subdirectory(examples)
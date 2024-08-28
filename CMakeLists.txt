cmake_minimum_required(VERSION 3.5)
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

set(TENSORFLOW_SOURCE_DIR "${CMAKE_SOURCE_DIR}/submodules/tensorflow")
message(STATUS "Tensorflow source dir: ${TENSORFLOW_SOURCE_DIR}")

add_subdirectory(
        "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# Include OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Source and include directories
SET(SRC_DIR src)
SET(INCLUDE_DIR tflite_inference_engine)

# Glob for source and header files
FILE(GLOB SRC_FILES ${SRC_DIR}/*.cpp)
FILE(GLOB INCLUDE_FILES ${INCLUDE_DIR}/*.hpp)

# Create a library from your source files
add_library(tflite_inference_engine_lib ${SRC_FILES} ${INCLUDE_FILES})
target_include_directories(tflite_inference_engine_lib PUBLIC ${INCLUDE_DIR})
target_link_libraries(tflite_inference_engine_lib ${OpenCV_LIBRARIES} tensorflow-lite)

# Create the main executable
add_executable(tflite_inference_engine ${SRC_FILES} ${INCLUDE_FILES})
target_link_libraries(tflite_inference_engine tflite_inference_engine_lib tensorflow-lite ${OpenCV_LIBRARIES})

add_subdirectory(tests)
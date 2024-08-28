set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

FILE(GLOB EXAMPLE_FILES "${CMAKE_SOURCE_DIR}/examples/example_*.cpp")

message(STATUS "CMake Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "------------- BUILDING EXAMPLES -------------")

foreach (EXAMPLE_FILE IN LISTS EXAMPLE_FILES)
    get_filename_component(EXAMPLE_FILE_WE ${EXAMPLE_FILE} NAME_WE)
    message("EXAMPLE NAME: ${EXAMPLE_FILE_WE}")
    add_executable(${EXAMPLE_FILE_WE} ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE_FILE_WE}.cpp)
    target_link_libraries(${EXAMPLE_FILE_WE} PUBLIC tflite_inference_engine_lib tensorflow-lite ${OpenCV_LIBRARIES})
endforeach ()
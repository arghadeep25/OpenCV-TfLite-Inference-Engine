# Add Google Test
add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/googletest ${CMAKE_BINARY_DIR}/submodules/googletest)

# Set test directory
set(TEST_DIR ${CMAKE_SOURCE_DIR}/tests)

# Glob all the test files (test_*.cpp) in the tests directory
file(GLOB TEST_FILES ${TEST_DIR}/*.cpp)

# Print the test files being included
message(STATUS "Test files: ${TEST_FILES}")

# Add test executable
add_executable(runTests ${TEST_FILES})

# Link test executable with Google Test and your library
target_link_libraries(runTests gtest gtest_main security_cam_lib tensorflow-lite ${OpenCV_LIBRARIES})

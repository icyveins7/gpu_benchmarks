cmake_minimum_required(VERSION 3.24)

project(
  "proj_containers"
DESCRIPTION "Custom containers"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(streams_example.cu PROPERTIES LANGUAGE CUDA)
add_executable(streams_example)

target_include_directories(streams_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  streams_example PRIVATE
  ${PROJECT_SOURCE_DIR}/streams_example.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/containers/streams.cu
)

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_remap")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

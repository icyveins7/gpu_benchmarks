cmake_minimum_required(VERSION 3.18)

project(
  "proj_containers"
DESCRIPTION "Custom containers"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(streams.cu PROPERTIES LANGUAGE CUDA)
add_executable(streams)

target_include_directories(streams
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  streams PRIVATE
  ${PROJECT_SOURCE_DIR}/streams.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/containers.cu
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

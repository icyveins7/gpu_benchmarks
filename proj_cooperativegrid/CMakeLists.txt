cmake_minimum_required(VERSION 3.24)

project(
  "proj_cooperativegrid"
DESCRIPTION "Cooperative grids"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(cooperativegridtest.cu PROPERTIES LANGUAGE CUDA)
add_executable(cooperativegridtest)

target_include_directories(cooperativegridtest
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  cooperativegridtest PRIVATE
  ${PROJECT_SOURCE_DIR}/cooperativegridtest.cu
)

# -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_cooperativegrid")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

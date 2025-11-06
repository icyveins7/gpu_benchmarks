cmake_minimum_required(VERSION 3.24)

project(
  "proj_basicops"
DESCRIPTION "Basic operations"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(basicops.cu PROPERTIES LANGUAGE CUDA)
add_executable(basicops)

target_include_directories(basicops
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  basicops PRIVATE
  ${PROJECT_SOURCE_DIR}/basicops.cu
)
# --------------------
set_source_files_properties(inflate.cu PROPERTIES LANGUAGE CUDA)
add_executable(inflate)

target_include_directories(inflate
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  inflate PRIVATE
  ${PROJECT_SOURCE_DIR}/inflate.cu
)

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_basicops")
#
#   # --------------------
#   add_executable(test_bitset test_bitset.cu)
#   target_link_libraries(test_bitset GTest::gtest_main)
#   gtest_discover_tests(test_bitset)
# endif()

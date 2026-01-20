cmake_minimum_required(VERSION 3.24)

project(
  "proj_floodfill"
  DESCRIPTION "Flood fill experiments"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(floodfillnpp.cu PROPERTIES LANGUAGE CUDA)
add_executable(floodfillnpp)

target_include_directories(floodfillnpp
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_link_libraries(
  floodfillnpp
  PRIVATE
  CUDA::nppc_static
  CUDA::nppif_static
  CUDA::culibos
)

target_sources(
  floodfillnpp PRIVATE
  ${PROJECT_SOURCE_DIR}/floodfillnpp.cu
)

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_fmamat")
#
#   # --------------------
#   add_executable(test_fmamat test_fmamat.cu)
#   target_link_libraries(test_fmamat GTest::gtest_main)
#   gtest_discover_tests(test_fmamat)
# endif()

cmake_minimum_required(VERSION 3.24)

project(
  "proj_tridiag"
  DESCRIPTION "Tridiagonal Linear Systems"
  LANGUAGES CXX CUDA)

# --------------------
add_executable(tridiag)
target_sources(
  tridiag PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tridiag.cu
)

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_windowccl")
#
#   # --------------------
#   set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test/test_wccl.cpp PROPERTIES LANGUAGE CUDA)
#   add_executable(test_wccl test/test_wccl.cpp src/wccl.cpp)
#   target_link_libraries(test_wccl GTest::gtest_main)
#   target_include_directories(
#     test_wccl PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#   )
#   gtest_discover_tests(test_wccl)
# endif()

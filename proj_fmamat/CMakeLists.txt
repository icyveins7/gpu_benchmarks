cmake_minimum_required(VERSION 3.24)

project(
  "proj_fmamat"
DESCRIPTION "FMA across a matrix"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(fmamat.cu PROPERTIES LANGUAGE CUDA)
add_executable(fmamat)

target_include_directories(fmamat
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  fmamat PRIVATE
  ${PROJECT_SOURCE_DIR}/fmamat.cu
)

# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_fmamat")

  # --------------------
  add_executable(test_fmamat test_fmamat.cu)
  target_link_libraries(test_fmamat GTest::gtest_main)
  gtest_discover_tests(test_fmamat)
endif()

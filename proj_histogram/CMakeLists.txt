cmake_minimum_required(VERSION 3.24)

project(
  "proj_histogram"
DESCRIPTION "Histograms"
  LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/../py2cpptest/unittest.cmake)

# --------------------
set_source_files_properties(hist.cu PROPERTIES LANGUAGE CUDA)
add_executable(hist)

target_include_directories(hist
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  hist PRIVATE
  ${PROJECT_SOURCE_DIR}/hist.cu
)

# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_histogram")

  # --------------------
  set_source_files_properties(test_hist.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(test_hist test_hist.cpp)
  target_include_directories(test_hist PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(test_hist PRIVATE GTest::gtest)
  py2cpptest_add_gtest(test_hist)
  py2cpptest_add_py(test_hist test_hist.py)

  gtest_discover_tests(test_hist)
endif()

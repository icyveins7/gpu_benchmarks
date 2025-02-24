cmake_minimum_required(VERSION 3.18)

project(
  "proj_histogram"
DESCRIPTION "Histograms"
  LANGUAGES CXX CUDA)

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
# if (GTEST_FOUND)
#   message("Enabling tests for proj_rng")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

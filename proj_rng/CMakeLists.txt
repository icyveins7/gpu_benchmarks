cmake_minimum_required(VERSION 3.18)

project(
  "proj_rng"
DESCRIPTION "Random number generators"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(thrustrng.cu PROPERTIES LANGUAGE CUDA)
add_executable(thrustrng)

target_include_directories(thrustrng
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  thrustrng PRIVATE
  ${PROJECT_SOURCE_DIR}/thrustrng.cu
)
# --------------------
set_source_files_properties(curandrng.cu PROPERTIES LANGUAGE CUDA)
add_executable(curandrng)

target_include_directories(curandrng
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  curandrng PRIVATE
  ${PROJECT_SOURCE_DIR}/curandrng.cu
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

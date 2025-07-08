cmake_minimum_required(VERSION 3.24)

project(
  "proj_selections"
DESCRIPTION "Selection kernels for slicing data from a larger array"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(selections.cu PROPERTIES LANGUAGE CUDA)
add_executable(selections)

target_include_directories(selections
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  selections PRIVATE
  ${PROJECT_SOURCE_DIR}/selections.cu
)
# --------------------
set_source_files_properties(conditioned_selections.cu PROPERTIES LANGUAGE CUDA)
add_executable(conditioned_selections)

target_include_directories(conditioned_selections
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  conditioned_selections PRIVATE
  ${PROJECT_SOURCE_DIR}/conditioned_selections.cu
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

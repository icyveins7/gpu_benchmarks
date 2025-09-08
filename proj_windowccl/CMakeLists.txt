cmake_minimum_required(VERSION 3.24)

project(
  "proj_windowccl"
  DESCRIPTION "Window CCL"
  LANGUAGES CXX CUDA)

# --------------------
add_executable(wccl_experiment)

target_include_directories(
  wccl_experiment PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(
  wccl_experiment PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/wccl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/wccl_experiment.cpp
)
# # --------------------
# set_source_files_properties(curandrng.cu PROPERTIES LANGUAGE CUDA)
# add_executable(curandrng)
#
# target_include_directories(curandrng
#                            PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#                            )
#
# target_sources(
#   curandrng PRIVATE
#   ${PROJECT_SOURCE_DIR}/curandrng.cu
# )

# -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_rng")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

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
if (GTEST_FOUND)
  message("Enabling tests for proj_windowccl")

  # --------------------
  add_executable(test_wccl test/test_wccl.cpp src/wccl.cpp)
  # target_link_libraries(test_wccl ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
  target_link_libraries(test_wccl GTest::gtest_main)
  target_include_directories(
    test_wccl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  gtest_discover_tests(test_wccl)
endif()

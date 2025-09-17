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
# --------------------
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/wccl_experiment_cuda.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/include/wccl_kernels.cuh PROPERTIES LANGUAGE CUDA)

add_executable(wccl_experiment_cuda)
target_include_directories(
  wccl_experiment_cuda
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_sources(
  wccl_experiment_cuda PRIVATE
  ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
)
# Add second one with a preprocessor flag
add_executable(wccl_experiment_cuda_atomicfreelocal)
target_compile_options(
  wccl_experiment_cuda_atomicfreelocal
  PRIVATE
  -DUSE_ATOMICFREE_LOCAL
)
target_include_directories(
  wccl_experiment_cuda_atomicfreelocal
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_sources(
  wccl_experiment_cuda_atomicfreelocal PRIVATE
  ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
)

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

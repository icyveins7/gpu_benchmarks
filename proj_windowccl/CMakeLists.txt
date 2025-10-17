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
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/include/wccl_hybrid.cuh PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/include/wccl_hybrid.cu PROPERTIES LANGUAGE CUDA)

add_executable(wccl_experiment_cuda)
target_include_directories(
  wccl_experiment_cuda
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(
  wccl_experiment_cuda
  PRIVATE
  -g
  # -G
  -lineinfo
)
target_sources(
  wccl_experiment_cuda PRIVATE
  ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
)
# Add extra one to swap between active site / window search
add_executable(wccl_experiment_cuda_useactivesitesinwindow)
target_include_directories(
  wccl_experiment_cuda_useactivesitesinwindow
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(
  wccl_experiment_cuda_useactivesitesinwindow
  PRIVATE
  -DUSE_ACTIVESITES_IN_WINDOW
  -g
  # -G
  -lineinfo
)
target_sources(
  wccl_experiment_cuda_useactivesitesinwindow PRIVATE
  ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
)

# # Add another one with a preprocessor flag for atomic free (DEPRECATED)
# add_executable(wccl_experiment_cuda_atomicfreelocal)
# target_compile_options(
#   wccl_experiment_cuda_atomicfreelocal
#   PRIVATE
#   -DUSE_ATOMICFREE_LOCAL
# )
# target_include_directories(
#   wccl_experiment_cuda_atomicfreelocal
#   PRIVATE
#   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#   ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# target_sources(
#   wccl_experiment_cuda_atomicfreelocal PRIVATE
#   ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
# )

# And another one also for neighbour chaining
add_executable(wccl_experiment_cuda_neighbourchainlocal)
target_compile_options(
  wccl_experiment_cuda_neighbourchainlocal
  PRIVATE
  -DUSE_NEIGHBOURCHAINER
  -g
  # -G
  -lineinfo
)
target_include_directories(
  wccl_experiment_cuda_neighbourchainlocal
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_sources(
  wccl_experiment_cuda_neighbourchainlocal PRIVATE
  ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
)

# And another one also for hybrid CPU-GPU neighbour chaining
add_executable(wccl_experiment_cuda_neighbourchainhybrid)
target_compile_options(
  wccl_experiment_cuda_neighbourchainhybrid
  PRIVATE
  -DUSE_NEIGHBOURCHAINERHYBRID
  -g
  # -G
  -lineinfo
)
target_include_directories(
  wccl_experiment_cuda_neighbourchainhybrid
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_sources(
  wccl_experiment_cuda_neighbourchainhybrid PRIVATE
  ${PROJECT_SOURCE_DIR}/src/wccl_experiment_cuda.cu
  ${PROJECT_SOURCE_DIR}/src/wccl_hybrid.cu
)


# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_windowccl")

  # --------------------
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test/test_wccl.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(test_wccl test/test_wccl.cpp src/wccl.cpp)
  target_link_libraries(test_wccl GTest::gtest_main)
  target_include_directories(
    test_wccl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  gtest_discover_tests(test_wccl)
endif()

cmake_minimum_required(VERSION 3.18)

project(
  "gpu_benchmarks"
  DESCRIPTION "A bunch of personal task-specific tests to measure GPU speedups"
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Add compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message("CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}")
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

# -------------------- Include individual project directories
add_subdirectory(proj_grid_polynomials)


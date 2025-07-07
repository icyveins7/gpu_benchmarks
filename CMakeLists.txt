cmake_minimum_required(VERSION 3.24)

project(
  "gpu_benchmarks"
  DESCRIPTION "A bunch of personal task-specific tests to measure GPU speedups"
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Add compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message("CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}")
endif()

# Default to native architecture alone
set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "Default to native CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# if(CMAKE_CUDA_ARCHITECTURES LESS 60)
#   set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
# endif()

# Enable warnings for GCC and Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wall -Wextra) # don't use Wpedantic for cuda code or you get tons of useless warnings
  # Enable warnings for MSVC
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/W4)
endif()

# Fixes for clangd
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# Common includes for all
include_directories(include/)

# Check if MatX projects should be built
if (BUILD_MATX_PROJECTS)
  message(STATUS "Building MatX projects")
else()
  message(STATUS "Not building MatX projects, enable via -DBUILD_MATX_PROJECTS=ON")
endif()

# -------------------- Add check for GoogleTest to see if it exists
find_package(GTest)
# If GTest found then enable tests, otherwise disable them
if(GTEST_FOUND)
  include(GoogleTest)
  enable_testing()
  message(STATUS "GoogleTest found, enabling tests")
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  message(STATUS "GoogleTest not found, disabling tests")
endif()


# -------------------- Include individual project directories
add_subdirectory(proj_grid_polynomials)
add_subdirectory(proj_remap)
add_subdirectory(proj_rng)
add_subdirectory(proj_fftshift)
add_subdirectory(proj_sumAndDownsampleMatrix)
add_subdirectory(proj_ffts)
add_subdirectory(proj_histogram)
add_subdirectory(proj_pinnedalloc)
add_subdirectory(proj_patternMatching)
add_subdirectory(proj_containers)
add_subdirectory(proj_fmamat)
add_subdirectory(proj_cropping)
add_subdirectory(proj_reductions)
add_subdirectory(proj_median)
add_subdirectory(proj_selections)


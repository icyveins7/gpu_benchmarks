cmake_minimum_required(VERSION 3.24)

# For windows, we use vcpkg for things like GTest,
# and there's a lot of issues with the default (which is dynamic linking)
if (WIN32)
  message("Using vcpkg from $ENV{VCPKG_ROOT}. If this is incorrect, please set it just before calling cmake.")
  # Need to convert it otherwise it complains about the backslashes
  cmake_path(SET VCPKG_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
  message("VCPKG_TOOLCHAIN_FILE: ${VCPKG_TOOLCHAIN_FILE}")
  set(CMAKE_TOOLCHAIN_FILE ${VCPKG_TOOLCHAIN_FILE})
#   set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "Force static build for windows targets")
#   set(VCPKG_HOST_TRIPLET "x64-windows-static" CACHE STRING "Force static build for host tools")
#   message("x64-windows-static is enforced for vcpkg - used for GTest. You must install gtest:x64-windows-static in order to properly build the tests on Windows.")
endif()
#
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

# Check if CUDA_VISIBLE_DEVICES is defined
if (NOT DEFINED ENV{CUDA_VISIBLE_DEVICES})
  message(WARNING
  "CUDA_VISIBLE_DEVICES not set, defaulting to 0. "
  "If you'd like to build for another GPU (or multiple GPU)'s compute capability, set this explicitly and rerun CMake.")
  set(cmake_cuda_devices 0)

else()
  set(cmake_cuda_devices $ENV{CUDA_VISIBLE_DEVICES})
endif()
message(STATUS "CUDA_VISIBLE_DEVICES set to ${cmake_cuda_devices}")

# Run nvidia-smi to try to get the native architecture
execute_process(
  COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i ${cmake_cuda_devices}
  OUTPUT_VARIABLE cmake_gpu_arch
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "\\." "" cmake_gpu_arch ${cmake_gpu_arch})
string(REPLACE "\n" ";" cmake_gpu_arch ${cmake_gpu_arch})
list(JOIN cmake_gpu_arch "," cmake_gpu_arch)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${cmake_gpu_arch}")


# Default to native architecture alone for the selected GPUs
# set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES ${cmake_gpu_arch})
message(STATUS "Default to native CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# if(CMAKE_CUDA_ARCHITECTURES LESS 60)
#   set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
# endif()

# Enable warnings for GCC and Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wall -Wextra) # don't use Wpedantic for cuda code or you get tons of useless warnings
  # Enable warnings for MSVC
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # add_compile_options(/W4) # this breaks the windows build, dont know why yet
endif()

# Fixes for clangd
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# Common includes for all
include_directories(
  include/
  cxxopts/include/
)

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

  # Save the absolute path to the generic main file in a variable
  set(GENERIC_TEST_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/test/test_main_generic.cu" CACHE INTERNAL "")
  if (WIN32)
    # A lot of nonsense warnings from gtest on windows..
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1394>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1388>)
  endif()
else()
  message(STATUS "GoogleTest not found, disabling tests")
endif()

# -------------------- Add some automatic detection of virtualenv's pybind11
execute_process(
  COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_python_result
)
if(NOT pybind11_python_result EQUAL 0)
  message(STATUS "Could not determine pybind11_DIR from Python (pybind11 may not be installed)")
  unset(pybind11_DIR)
endif()
message(STATUS "Automatically detected pybind11_DIR: ${pybind11_DIR}")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG)
if (pybind11_FOUND)
  message(STATUS "pybind11 found, enabling pybind11-dependent targets")
else()
  message(STATUS "pybind11 not found, disabling pybind11-dependent targets")
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
add_subdirectory(proj_outlierHandling)
add_subdirectory(proj_windowccl)
add_subdirectory(proj_basicops)
add_subdirectory(proj_cooperativegrid)
add_subdirectory(proj_sats)
add_subdirectory(proj_mempools)
add_subdirectory(proj_floodfill)
add_subdirectory(proj_slipups)
add_subdirectory(proj_cudagraphs)
add_subdirectory(proj_tridiag)


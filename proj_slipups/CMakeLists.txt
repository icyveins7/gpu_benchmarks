cmake_minimum_required(VERSION 3.24)

project(
  "proj_slipups"
  DESCRIPTION "A record of things that may go wrong"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(printf_in_kernels.cu PROPERTIES LANGUAGE CUDA)
add_executable(printf_in_kernels)

target_include_directories(printf_in_kernels
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  printf_in_kernels PRIVATE
  ${PROJECT_SOURCE_DIR}/printf_in_kernels.cu
)


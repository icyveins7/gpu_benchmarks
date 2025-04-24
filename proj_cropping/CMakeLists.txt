cmake_minimum_required(VERSION 3.24)

project(
  "proj_cropping"
DESCRIPTION "Cropping Windows"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(cropping.cu PROPERTIES LANGUAGE CUDA)
add_executable(cropping)

target_include_directories(cropping
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  cropping PRIVATE
  ${PROJECT_SOURCE_DIR}/cropping.cu
)


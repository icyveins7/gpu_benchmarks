cmake_minimum_required(VERSION 3.24)

project(
  "proj_reductions"
DESCRIPTION "Reductions"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(blkreductions.cu PROPERTIES LANGUAGE CUDA)
add_executable(blkreductions)

target_include_directories(blkreductions
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  blkreductions PRIVATE
  ${PROJECT_SOURCE_DIR}/blkreductions.cu
)

# --------------------
set_source_files_properties(devicereduction.cu PROPERTIES LANGUAGE CUDA)
add_executable(devicereduction)

target_include_directories(devicereduction
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  devicereduction PRIVATE
  ${PROJECT_SOURCE_DIR}/devicereduction.cu
)


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


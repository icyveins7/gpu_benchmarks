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

# --------------------
set_source_files_properties(thrustreduction.cu PROPERTIES LANGUAGE CUDA)
add_executable(thrustreduction)

target_include_directories(thrustreduction
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  thrustreduction PRIVATE
  ${PROJECT_SOURCE_DIR}/thrustreduction.cu
)

# --------------------
set_source_files_properties(cub_vs_cg.cu PROPERTIES LANGUAGE CUDA)
add_executable(cub_vs_cg)

target_include_directories(cub_vs_cg
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  cub_vs_cg PRIVATE
  ${PROJECT_SOURCE_DIR}/cub_vs_cg.cu
)

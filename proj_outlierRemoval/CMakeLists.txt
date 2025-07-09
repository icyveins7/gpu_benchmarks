cmake_minimum_required(VERSION 3.24)

project(
  "proj_outlierRemoval"
DESCRIPTION "Outlier Removal"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(outlierRemoval.cu PROPERTIES LANGUAGE CUDA)
add_executable(outlierRemoval)

target_include_directories(outlierRemoval
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_sources(
    outlierRemoval PRIVATE
    ${PROJECT_SOURCE_DIR}/outlierRemoval.cu
)

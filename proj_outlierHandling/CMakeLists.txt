cmake_minimum_required(VERSION 3.24)

project(
  "proj_outlierHandling"
DESCRIPTION "Outlier Handling"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(outlier.cu PROPERTIES LANGUAGE CUDA)
add_executable(outlier)

target_include_directories(outlier
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_sources(
    outlier PRIVATE
    ${PROJECT_SOURCE_DIR}/outlier.cu
)

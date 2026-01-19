cmake_minimum_required(VERSION 3.24)

project(
  "proj_basicops"
DESCRIPTION "Basic operations"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(basicops.cu PROPERTIES LANGUAGE CUDA)
add_executable(basicops)

target_include_directories(basicops
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  basicops PRIVATE
  ${PROJECT_SOURCE_DIR}/basicops.cu
)
# --------------------
set_source_files_properties(inflate.cu PROPERTIES LANGUAGE CUDA)
add_executable(inflate)

target_include_directories(inflate
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  inflate PRIVATE
  ${PROJECT_SOURCE_DIR}/inflate.cu
)
# --------------------
set_source_files_properties(deflate.cu PROPERTIES LANGUAGE CUDA)
add_executable(deflate)

target_include_directories(deflate
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  deflate PRIVATE
  ${PROJECT_SOURCE_DIR}/deflate.cu
)
# --------------------
set_source_files_properties(run_length_encode.cu PROPERTIES LANGUAGE CUDA)
add_executable(run_length_encode)

target_include_directories(run_length_encode
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  run_length_encode PRIVATE
  ${PROJECT_SOURCE_DIR}/run_length_encode.cu
)
# --------------------
set_source_files_properties(transpose.cu PROPERTIES LANGUAGE CUDA)
add_executable(transpose)

target_include_directories(transpose
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  transpose PRIVATE
  ${PROJECT_SOURCE_DIR}/transpose.cu
)
# --------------------
set_source_files_properties(cub_using_device_length.cu PROPERTIES LANGUAGE CUDA)
add_executable(cub_using_device_length)

target_include_directories(cub_using_device_length
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

set_property(TARGET cub_using_device_length PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_sources(
  cub_using_device_length PRIVATE
  ${PROJECT_SOURCE_DIR}/cub_using_device_length.cu
)

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_basicops")
#
#   # --------------------
#   add_executable(test_bitset test_bitset.cu)
#   target_link_libraries(test_bitset GTest::gtest_main)
#   gtest_discover_tests(test_bitset)
# endif()

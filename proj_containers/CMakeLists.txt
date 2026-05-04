cmake_minimum_required(VERSION 3.24)

project(
  "proj_containers"
DESCRIPTION "Custom containers"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(streams_example.cu PROPERTIES LANGUAGE CUDA)
add_executable(streams_example)

target_include_directories(streams_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  streams_example PRIVATE
  ${PROJECT_SOURCE_DIR}/streams_example.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/containers/streams.cu
)
# --------------------
set_source_files_properties(bitset_speed.cu PROPERTIES LANGUAGE CUDA)
add_executable(bitset_speed)

target_include_directories(bitset_speed
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  bitset_speed PRIVATE
  ${PROJECT_SOURCE_DIR}/bitset_speed.cu
)

# --------------------
set_source_files_properties(${PROJECT_SOURCE_DIR}/sort.cu PROPERTIES LANGUAGE CUDA)
add_executable(sort)

target_include_directories(sort
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  sort PRIVATE
  ${PROJECT_SOURCE_DIR}/sort.cu
)

# --------------------
set_source_files_properties(${PROJECT_SOURCE_DIR}/deviceselect.cu PROPERTIES LANGUAGE CUDA)
add_executable(deviceselect)

target_include_directories(deviceselect
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  deviceselect PRIVATE
  ${PROJECT_SOURCE_DIR}/deviceselect.cu
)

# --------------------
set_source_files_properties(${PROJECT_SOURCE_DIR}/stream_ordered_alloc.cu PROPERTIES LANGUAGE CUDA)
add_executable(stream_ordered_alloc)

target_include_directories(stream_ordered_alloc
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  stream_ordered_alloc PRIVATE
  ${PROJECT_SOURCE_DIR}/stream_ordered_alloc.cu
  ${CMAKE_SOURCE_DIR}/src/containers/streams.cu
)

# 
add_executable(stream_ordered_alloc_with_memcpy)
target_include_directories(stream_ordered_alloc_with_memcpy
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )
target_compile_definitions(stream_ordered_alloc_with_memcpy PRIVATE WITH_MEMCPY)
target_sources(
  stream_ordered_alloc_with_memcpy PRIVATE
  ${PROJECT_SOURCE_DIR}/stream_ordered_alloc.cu
  ${CMAKE_SOURCE_DIR}/src/containers/streams.cu
)

# --------------------
add_executable(stream_ordered_alloc_comparison_to_devicevec)
target_include_directories(stream_ordered_alloc_comparison_to_devicevec
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )
target_sources(
  stream_ordered_alloc_comparison_to_devicevec PRIVATE
  ${PROJECT_SOURCE_DIR}/stream_ordered_alloc_comparison.cu
  ${CMAKE_SOURCE_DIR}/src/containers/streams.cu
)


# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_containers")

  # --------------------
  add_executable(test_bitset test_bitset.cu)
  target_link_libraries(test_bitset GTest::gtest_main)
  gtest_discover_tests(test_bitset)

  # --------------------
  add_executable(test_image test_image.cu)
  target_link_libraries(test_image GTest::gtest_main)
  gtest_discover_tests(test_image)
endif()

cmake_minimum_required(VERSION 3.24)

project(
  "proj_oneblock"
DESCRIPTION "One block tests"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/sort.cu PROPERTIES LANGUAGE CUDA)
add_executable(oneblocksort)

target_include_directories(oneblocksort
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  oneblocksort PRIVATE
  ${PROJECT_SOURCE_DIR}/src/sort.cu
)

# --------------------
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/deviceselect.cu PROPERTIES LANGUAGE CUDA)
add_executable(oneblockselect)

target_include_directories(oneblockselect
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  oneblockselect PRIVATE
  ${PROJECT_SOURCE_DIR}/src/deviceselect.cu
)

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_remap")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

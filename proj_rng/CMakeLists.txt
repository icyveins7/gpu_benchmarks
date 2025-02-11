cmake_minimum_required(VERSION 3.18)

project(
  "proj_rng"
DESCRIPTION "Random number generators"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
add_executable(rng)

target_include_directories(rng
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  rng PRIVATE
  ${PROJECT_SOURCE_DIR}/main.cu
)

# -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_rng")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

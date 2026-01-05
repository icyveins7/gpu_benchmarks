cmake_minimum_required(VERSION 3.24)

project(
  "proj_sats"
DESCRIPTION "Summed area tables"
  LANGUAGES CXX CUDA)

# --------------------
set_source_files_properties(sats.cu PROPERTIES LANGUAGE CUDA)
add_executable(sats)

target_include_directories(sats
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  sats PRIVATE
  ${PROJECT_SOURCE_DIR}/sats.cu
)

# --------------------


# -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_remap")
#
#   # --------------------
#   add_executable(test_remap test_remap.cu)
#   target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
#   gtest_discover_tests(test_remap)
# endif()

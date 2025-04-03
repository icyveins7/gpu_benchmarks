cmake_minimum_required(VERSION 3.24)

project(
  "proj_patternMatching"
DESCRIPTION "Pattern matching"
  LANGUAGES CXX CUDA)

# include(${CMAKE_CURRENT_SOURCE_DIR}/../py2cpptest/unittest.cmake)

# --------------------
set_source_files_properties(matching.cu PROPERTIES LANGUAGE CUDA)
add_executable(matching)

target_include_directories(matching
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_sources(
    matching PRIVATE
    ${PROJECT_SOURCE_DIR}/matching.cu
)
# --------------------
# if (BUILD_MATX_PROJECTS)
#   set_source_files_properties(matcher_matx.cu PROPERTIES LANGUAGE CUDA)
#   add_executable(matcher_matx)
#
#   target_include_directories(matcher_matx
#     PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#   )
#
#   target_sources(
#     matcher_matx PRIVATE
#     ${PROJECT_SOURCE_DIR}/matcher_matx.cu
#   )
# endif()

# # -------------------- Tests
# if (GTEST_FOUND)
#   message("Enabling tests for proj_histogram")
#
#   # --------------------
#   set_source_files_properties(test_hist.cpp PROPERTIES LANGUAGE CUDA)
#   add_executable(test_hist test_hist.cpp)
#   target_include_directories(test_hist PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#   target_link_libraries(test_hist PRIVATE GTest::gtest)
#   py2cpptest_add_gtest(test_hist)
#   py2cpptest_add_py(test_hist test_hist.py)
#
#   gtest_discover_tests(test_hist)
# endif()

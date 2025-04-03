cmake_minimum_required(VERSION 3.24)

project(
  "proj_pinnedalloc"
DESCRIPTION "Pinned allocators"
  LANGUAGES CXX CUDA)

# include(${CMAKE_CURRENT_SOURCE_DIR}/../py2cpptest/unittest.cmake)

# --------------------
set_source_files_properties(thrustpinned.cu PROPERTIES LANGUAGE CUDA)
add_executable(thrustpinned)

target_include_directories(thrustpinned
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  thrustpinned PRIVATE
  ${PROJECT_SOURCE_DIR}/thrustpinned.cu
)

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

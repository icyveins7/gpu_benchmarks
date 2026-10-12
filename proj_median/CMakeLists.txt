cmake_minimum_required(VERSION 3.24)

project(
  "proj_median"
DESCRIPTION "Kernels to perform medians"
  LANGUAGES CXX CUDA)

# --------------------
if(NOT WIN32)
set_source_files_properties(medianExample.cu PROPERTIES LANGUAGE CUDA)
add_executable(medianExample)

target_compile_options(medianExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_include_directories(medianExample
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  medianExample PRIVATE
  ${PROJECT_SOURCE_DIR}/medianExample.cu
)
endif()

# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_median")

  add_executable(medianTest2d medianTest2d.cu ${GENERIC_TEST_MAIN})
  target_link_libraries(medianTest2d GTest::gtest)
  gtest_discover_tests(medianTest2d)
endif()

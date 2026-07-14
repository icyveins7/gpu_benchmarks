cmake_minimum_required(VERSION 3.24)

project(
  "proj_tridiag"
  DESCRIPTION "Tridiagonal Linear Systems"
  LANGUAGES CXX CUDA)

# --------------------
add_executable(tridiag)
target_sources(
  tridiag PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tridiag.cu
)

add_executable(tridiag_float)
target_sources(
  tridiag_float PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/tridiag.cu
)
target_compile_definitions(tridiag_float PRIVATE USE_FLOAT)

# --------------------
add_executable(splines)
target_sources(
  splines PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/splines.cu
)

# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_tridiag")

  add_executable(test_tridiag
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tridiag.cu
    ${GENERIC_TEST_MAIN}
  )
  target_link_libraries(test_tridiag GTest::gtest)
  gtest_discover_tests(test_tridiag)
endif()

cmake_minimum_required(VERSION 3.24)

project(
  "proj_remap"
DESCRIPTION "Bilinear interpolation remapping"
  LANGUAGES CXX CUDA)

if (UNIX)
  find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  find_library(NPPIG_LIBRARY nppig ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

  # find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  # find_library(NPPIF_LIBRARY nppif ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  # find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  # find_library(NPPISU_LIBRARY nppisu ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  # find_library(CULIBOS culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif (UNIX)

# if (MSVC OR WIN32 OR MSYS)
#   MESSAGE( STATUS "CMAKE_CUDA_COMPILER:         " ${CMAKE_CUDA_COMPILER} )
#   set(CUDA_LIBRARY_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64")
#   set(CUDART_LIBRARY ${CUDA_LIBRARY_PATH}/cudart.lib)
#   set(NPPIF_LIBRARY ${CUDA_LIBRARY_PATH}/nppif.lib)
#   set(NPPC_LIBRARY ${CUDA_LIBRARY_PATH}/nppc.lib)
#   set(NPPISU_LIBRARY ${CUDA_LIBRARY_PATH}/nppisu.lib)
#   set(CULIBOS ${CUDA_LIBRARY_PATH}/culibos.lib)
# endif()
# --------------------
set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
add_executable(remap)

target_include_directories(remap
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  remap PRIVATE
  ${PROJECT_SOURCE_DIR}/main.cu
)

# --------------------
set_source_files_properties(check_npp_edges.cu PROPERTIES LANGUAGE CUDA)
add_executable(check_npp_edges)

target_include_directories(check_npp_edges
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  check_npp_edges PRIVATE
  ${PROJECT_SOURCE_DIR}/check_npp_edges.cu
)
target_link_libraries(check_npp_edges ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY})

# --------------------
set_source_files_properties(downsample.cu PROPERTIES LANGUAGE CUDA)
add_executable(downsample)

target_include_directories(downsample
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  downsample PRIVATE
  ${PROJECT_SOURCE_DIR}/downsample.cu
)

# -------------------- Tests
if (GTEST_FOUND)
  message("Enabling tests for proj_remap")

  # --------------------
  add_executable(test_remap test_remap.cu)
  target_link_libraries(test_remap ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} GTest::gtest_main)
  gtest_discover_tests(test_remap)
endif()

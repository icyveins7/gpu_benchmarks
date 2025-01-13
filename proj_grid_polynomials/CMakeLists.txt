cmake_minimum_required(VERSION 3.18)

project(
  "proj_grid_polynomials"
DESCRIPTION "Polynomial computation over a grid"
  LANGUAGES CXX CUDA)


# --------------------
set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
add_executable(grid_polynomials)

target_include_directories(grid_polynomials
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

target_sources(
  grid_polynomials PRIVATE
  ${PROJECT_SOURCE_DIR}/main.cu
)

# target_link_libraries(1d_r2c_c2r_example PRIVATE CUDA::cufft CUDA::cudart)

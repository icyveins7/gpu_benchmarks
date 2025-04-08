#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include "../include/pinnedalloc.cuh"
#include "../include/sharedmem.cuh"

template <typename Tcalc, typename Tx, typename Tm, typename Tc, typename Ty>
void validate_fmamat_columns(
  const thrust::pinned_host_vector<Tx> &x,
  const int rows, const int columns,
  const thrust::pinned_host_vector<Tm> &m,
  const thrust::pinned_host_vector<Tc> &c,
  thrust::pinned_host_vector<Ty> &y
){
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j)
      y[i * columns + j] = (Tcalc)m[j] * (Tcalc)x[i * columns + j] + (Tcalc)c[j];
}


template <typename Tcalc, typename Tx, typename Tm, typename Tc, typename Ty>
__global__ void naive_fmamat_columns_kernel(
  const Tx *x,
  const int rows, const int columns,
  const Tm *m,
  const Tc *c,
  Ty *y
){
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < rows * columns; t += blockDim.x * gridDim.x)
    y[t] = (Tcalc)m[t % columns] * (Tcalc)x[t] + (Tcalc)c[t % columns];
}


template <typename Tcalc, typename Tx, typename Tm, typename Tc, typename Ty>
__global__ void shared_fmamat_columns_kernel(
  const Tx *x,
  const int rows, const int columns,
  const Tm *m,
  const Tc *c,
  Ty *y
){
  // Assumption: Each block does 32 columns. Larger blocks simply do more rows.
  // i.e. all kernels are (32, N)
  if (blockDim.x != 32)
    return;

  // The row and column indices for each thread (for src x and dst y)
  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int oIdx = rowIdx * columns + colIdx; // same as input index

  // Load our 32 columns of m and c
  __shared__ Tcalc s_m[32];
  __shared__ Tcalc s_c[32];
  for (int t = threadIdx.x; t < 32; t += blockDim.x){
    if (colIdx < columns){
      s_m[t] = m[colIdx];
      s_c[t] = c[colIdx];
    }
  }
  __syncthreads();

  // Computations for the block
  if (rowIdx < rows && colIdx < columns)
  {
    y[oIdx] = (Tcalc)s_m[threadIdx.x] * (Tcalc)x[oIdx] + (Tcalc)s_c[threadIdx.x];
  }

}

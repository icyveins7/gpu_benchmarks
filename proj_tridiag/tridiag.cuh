#pragma once

#include <sharedmem.cuh>
#include <thrust/device_vector.h>

#include <cstdint>

namespace cutridiag {

/*
 * Tridiagonal linear system solvers.
 *
 * NOTE: The Thomas algorithm cannot be solved via prefix scans because it is
 * not inherently associative i.e. the dependencies cannot be split up and
 * recombined simply. As such the only real way is to resort to the typical
 * Parallel Cyclic Reduction (or similar) technique.
 */

/**
 * @brief Parameters for PCR tridiagonal solver.
 * buf0 and buf1 are double-buffers for the reduction, each containing
 * a, b, c, rhs arrays of length >= length. These may reside in global
 * or shared memory. The output is written to u.
 *
 * @tparam T Type of data, must be floating point.
 */
template <typename T> struct tridiag_pcr_params {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating point type");
  T *a; // we leave these non-const so that other kernels can fill these in
  T *b; // e.g. splines will calculate these coefficients
  T *c;
  T *rhs;
  // Double buffers for the reduction (read/write)
  T *buf0_a, *buf0_b, *buf0_c, *buf0_rhs;
  T *buf1_a, *buf1_b, *buf1_c, *buf1_rhs;
  T *u; // output
  size_t length;
};

/**
 * @brief Helper that allocates the 8 double-buffer arrays in global memory
 * for the PCR solver.
 *
 * @tparam T Type of data, must be floating point.
 */
template <typename T> struct TridiagPCRWorkspace {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating point type");

  thrust::device_vector<T> buf0_a, buf0_b, buf0_c, buf0_rhs;
  thrust::device_vector<T> buf1_a, buf1_b, buf1_c, buf1_rhs;

  TridiagPCRWorkspace() = default;

  TridiagPCRWorkspace(int num_rows, int stride_elements_per_row) {
    resize(num_rows, stride_elements_per_row);
  }

  void resize(int num_rows, int stride_elements_per_row) {
    size_t total = (size_t)num_rows * stride_elements_per_row;
    buf0_a.resize(total);
    buf0_b.resize(total);
    buf0_c.resize(total);
    buf0_rhs.resize(total);
    buf1_a.resize(total);
    buf1_b.resize(total);
    buf1_c.resize(total);
    buf1_rhs.resize(total);
  }

  T *buf0_a_ptr() { return thrust::raw_pointer_cast(buf0_a.data()); }
  T *buf0_b_ptr() { return thrust::raw_pointer_cast(buf0_b.data()); }
  T *buf0_c_ptr() { return thrust::raw_pointer_cast(buf0_c.data()); }
  T *buf0_rhs_ptr() { return thrust::raw_pointer_cast(buf0_rhs.data()); }
  T *buf1_a_ptr() { return thrust::raw_pointer_cast(buf1_a.data()); }
  T *buf1_b_ptr() { return thrust::raw_pointer_cast(buf1_b.data()); }
  T *buf1_c_ptr() { return thrust::raw_pointer_cast(buf1_c.data()); }
  T *buf1_rhs_ptr() { return thrust::raw_pointer_cast(buf1_rhs.data()); }

  /**
   * @brief Returns the dynamic shared memory bytes needed for the shmem PCR
   * variant, given a system of length n.
   *
   * Requires 8 arrays of n elements each (2 buffers x 4 coefficients).
   * For double: 64 * n bytes. For float: 32 * n bytes.
   *
   * NOTE: n should be the stride_elements_per_row (i.e. the maximum number
   * of elements in any row), since the shmem is carved once and reused
   * across all rows processed by a block.
   *
   * NOTE: Default shmem limit is typically 48 KB (768 doubles, 1536 floats).
   * For larger n, use cudaFuncSetAttribute with
   * cudaFuncAttributeMaxDynamicSharedMemorySize to opt in to extended shmem
   * (up to ~100-164 KB depending on GPU architecture).
   */
  static constexpr size_t requiredShmemBytes(int n) {
    return 8 * n * sizeof(T);
  }
};

/**
 * @brief PCR tridiagonal solver within a single block.
 * Uses block-stride loops, so length may exceed blockDim.x.
 * The caller must provide two sets of buffers (buf0, buf1) in the params
 * for double-buffering during the reduction steps.
 *
 * @tparam T Type of data, see tridiag_pcr_params
 * @param params Parameters for PCR tridiagonal linear system
 */
template <typename T>
__device__ void tridiag_blockwide_pcr(tridiag_pcr_params<T> &params) {

  int n = params.length;

  // Copy inputs into buf0
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    params.buf0_a[i] = params.a[i];
    params.buf0_b[i] = params.b[i];
    params.buf0_c[i] = params.c[i];
    params.buf0_rhs[i] = params.rhs[i];
  }
  __syncthreads();

  // Set up read/write buffer pointers
  T *aR = params.buf0_a, *bR = params.buf0_b;
  T *cR = params.buf0_c, *rR = params.buf0_rhs;
  T *aW = params.buf1_a, *bW = params.buf1_b;
  T *cW = params.buf1_c, *rW = params.buf1_rhs;

  // Loop over neighbour distance
  // Initial distance is 1 i.e. the starting tridiagonal band
  for (int s = 1; s < n; s <<= 1) {
    // Block-stride: each thread handles multiple rows if n > blockDim.x
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      // left neighbour
      int lo = i - s;
      // right neighbour
      int hi = i + s;

      // Neighbour check must ensure it 'exists'
      T a_lo = (lo >= 0) ? aR[lo] : T(0);
      T b_lo =
          (lo >= 0) ? bR[lo] : T(1); // b gets divided, so use 1 as identity
      T c_lo = (lo >= 0) ? cR[lo] : T(0);
      T r_lo = (lo >= 0) ? rR[lo] : T(0);

      // Same 'existence' check
      T a_hi = (hi < n) ? aR[hi] : T(0);
      T b_hi = (hi < n) ? bR[hi] : T(1); // b gets divided, so use 1 as identity
      T c_hi = (hi < n) ? cR[hi] : T(0);
      T r_hi = (hi < n) ? rR[hi] : T(0);

      // 'New' coefficients
      // Add redundancy for direct setting of alpha/gamma
      // so that even if a[0] and c[n-1] are set, the correct value is still
      // computed
      T alpha = (lo >= 0) ? -aR[i] / b_lo : T(0);
      T gamma = (hi < n) ? -cR[i] / b_hi : T(0);

      aW[i] = alpha * a_lo;
      bW[i] = bR[i] + alpha * c_lo + gamma * a_hi;
      cW[i] = gamma * c_hi;
      rW[i] = rR[i] + alpha * r_lo + gamma * r_hi;
    }
    __syncthreads();

    // Swap buffers
    T *tmp;
    tmp = aR;
    aR = aW;
    aW = tmp;

    tmp = bR;
    bR = bW;
    bW = tmp;

    tmp = cR;
    cR = cW;
    cW = tmp;

    tmp = rR;
    rR = rW;
    rW = tmp;
  }

  // All rows are independent: x[i] = r[i] / b[i]
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    params.u[i] = rR[i] / bR[i];
  }
  __syncthreads();
}

template <typename T>
__global__ void tridiag_blockwise_pcr_kernel(
    const T *a, const T *b, const T *c, const T *rhs, T *u, T *buf0_a,
    T *buf0_b, T *buf0_c, T *buf0_rhs, T *buf1_a, T *buf1_b, T *buf1_c,
    T *buf1_rhs, const size_t *num_elements_in_row,
    const int stride_elements_per_row, const int num_rows) {

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    tridiag_pcr_params<T> params = {const_cast<T *>(&a[off]),
                                    const_cast<T *>(&b[off]),
                                    const_cast<T *>(&c[off]),
                                    const_cast<T *>(&rhs[off]),
                                    &buf0_a[off],
                                    &buf0_b[off],
                                    &buf0_c[off],
                                    &buf0_rhs[off],
                                    &buf1_a[off],
                                    &buf1_b[off],
                                    &buf1_c[off],
                                    &buf1_rhs[off],
                                    &u[off],
                                    num_elements_in_row[blk]};
    tridiag_blockwide_pcr<T>(params);
  }
}

// NOTE: early measurements show that for doubles, this has effectively the same
// performance as the non-shmem variant above, likely due to exceedingly low
// occupancy
template <typename T>
__global__ void tridiag_blockwise_pcr_shmem_kernel(
    const T *a, const T *b, const T *c, const T *rhs, T *u,
    const size_t *num_elements_in_row, const int stride_elements_per_row,
    const int num_rows) {

  SharedMemory<T> smem;
  T *shmem = smem.getPointer();

  // Carve out 8 arrays from dynamic shmem, using stride as the max length
  T *buf0_a = shmem;
  T *buf0_b = buf0_a + stride_elements_per_row;
  T *buf0_c = buf0_b + stride_elements_per_row;
  T *buf0_rhs = buf0_c + stride_elements_per_row;
  T *buf1_a = buf0_rhs + stride_elements_per_row;
  T *buf1_b = buf1_a + stride_elements_per_row;
  T *buf1_c = buf1_b + stride_elements_per_row;
  T *buf1_rhs = buf1_c + stride_elements_per_row;

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    int n = num_elements_in_row[blk];

    tridiag_pcr_params<T> params = {const_cast<T *>(&a[off]),
                                    const_cast<T *>(&b[off]),
                                    const_cast<T *>(&c[off]),
                                    const_cast<T *>(&rhs[off]),
                                    buf0_a,
                                    buf0_b,
                                    buf0_c,
                                    buf0_rhs,
                                    buf1_a,
                                    buf1_b,
                                    buf1_c,
                                    buf1_rhs,
                                    &u[off],
                                    (size_t)n};
    tridiag_blockwide_pcr<T>(params);
  }
}

} // end namespace cutridiag

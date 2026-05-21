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
 * @brief Double-buffer workspace for the PCR reduction.
 * buf0 and buf1 each contain a, b, c, rhs arrays of length >= length.
 * These may reside in global or shared memory.
 *
 * For global memory workspaces, see TriDiagPCRWorkspace below for a helpful
 * container.
 *
 * buf0 should be populated at the start. This is left to the caller to do so
 * appropriately.
 *
 * @tparam T Type of data, must be floating point.
 */
template <typename T> struct tridiag_pcr_scratch {
  static_assert(std::is_floating_point<T>::value,
                "T must be a floating point type");
  T *buf0_a, *buf0_b, *buf0_c, *buf0_rhs;
  T *buf1_a, *buf1_b, *buf1_c, *buf1_rhs;
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
 * @brief Core PCR reduction loop for a single block.
 * NOTE: Assumes buf0 already contains the initial a, b, c, rhs data.
 * Onus is on the caller to populate this beforehand. See tridiag_pcr_params.
 *
 * Performs the reduction in-place across buf0/buf1 (double-buffering).
 *
 * On return, rFinal[i]/bFinal[i] gives the solution x[i]. The caller is
 * responsible for performing this division, e.g.:
 *   out[i] = rFinal[i] / bFinal[i];   // standard solve
 *   out[i] -= k * rFinal[i] / bFinal[i]; // accumulate (Sherman-Morrison)
 *
 * Since the output is the caller's responsibility, __syncthreads() is *NOT*
 * automatically executed at the end, so the caller must syncthreads if needed.
 *
 * @param buf0_a, buf0_b, buf0_c, buf0_rhs  First buffer set
 * @param buf1_a, buf1_b, buf1_c, buf1_rhs  Second buffer set
 * @param n        Number of equations
 * @param rFinal   [out] pointer to the final rhs buffer after reduction
 * @param bFinal   [out] pointer to the final b buffer after reduction
 */
template <typename T>
__device__ __forceinline__ void
pcr_reduce_blockwide(T *buf0_a, T *buf0_b, T *buf0_c, T *buf0_rhs, T *buf1_a,
                     T *buf1_b, T *buf1_c, T *buf1_rhs, int n, T *&rFinal,
                     T *&bFinal) {
  T *aR = buf0_a, *bR = buf0_b;
  T *cR = buf0_c, *rR = buf0_rhs;
  T *aW = buf1_a, *bW = buf1_b;
  T *cW = buf1_c, *rW = buf1_rhs;

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

  rFinal = rR;
  bFinal = bR;
}

/**
 * @brief PCR tridiagonal solver within a single block.
 * Uses block-stride loops, so length may exceed blockDim.x.
 * Copies the input arrays (a,b,c,rhs) into buf0, then performs PCR.
 *
 * @tparam T Type of data, see tridiag_pcr_params
 * @param params PCR double-buffer workspace
 * @param a, b, c, rhs  Input tridiagonal coefficients (length >= n)
 * @param out    Output array of length >= n
 */
template <typename T>
__device__ void tridiag_blockwide_pcr(tridiag_pcr_scratch<T> &params,
                                      const T *a, const T *b, const T *c,
                                      const T *rhs, T *out) {

  int n = params.length;

  // Copy inputs into buf0
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    params.buf0_a[i] = a[i];
    params.buf0_b[i] = b[i];
    params.buf0_c[i] = c[i];
    params.buf0_rhs[i] = rhs[i];
  }
  __syncthreads();

  T *rFinal, *bFinal;
  pcr_reduce_blockwide<T>(params.buf0_a, params.buf0_b, params.buf0_c,
                          params.buf0_rhs, params.buf1_a, params.buf1_b,
                          params.buf1_c, params.buf1_rhs, n, rFinal, bFinal);

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = rFinal[i] / bFinal[i];
  }

  // We syncthreads here since this function may be called repeatedly
  __syncthreads();
}

/**
 * @brief Standard tridiagonal solver kernel (global memory variant).
 *
 * @example For a 4x4 system:
 *
 *   | b0  c0   0   0 |       a[] = { 0,  a1, a2, a3 }
 *   | a1  b1  c1   0 |       b[] = { b0, b1, b2, b3 }
 *   |  0  a2  b2  c2 |       c[] = { c0, c1, c2, 0  }
 *   |  0   0  a3  b3 |
 */
template <typename T>
__global__ void tridiag_blockwise_pcr_kernel(
    const T *a, const T *b, const T *c, const T *rhs, T *u, T *buf0_a,
    T *buf0_b, T *buf0_c, T *buf0_rhs, T *buf1_a, T *buf1_b, T *buf1_c,
    T *buf1_rhs, const size_t *num_elements_in_row,
    const int stride_elements_per_row, const int num_rows) {

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    tridiag_pcr_scratch<T> params = {
        &buf0_a[off],   &buf0_b[off],   &buf0_c[off],
        &buf0_rhs[off], &buf1_a[off],   &buf1_b[off],
        &buf1_c[off],   &buf1_rhs[off], num_elements_in_row[blk]};
    tridiag_blockwide_pcr<T>(params, &a[off], &b[off], &c[off], &rhs[off],
                             &u[off]);
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

    tridiag_pcr_scratch<T> params = {buf0_a, buf0_b, buf0_c,   buf0_rhs, buf1_a,
                                     buf1_b, buf1_c, buf1_rhs, (size_t)n};
    tridiag_blockwide_pcr<T>(params, &a[off], &b[off], &c[off], &rhs[off],
                             &u[off]);
  }
}

/**
 * @brief Cyclic tridiagonal solver via Sherman-Morrison decomposition.
 * Performs two PCR solves internally. The final combined result is written
 * to out. The second solve's result is read directly from the double buffers,
 * avoiding the need for a second output array.
 *
 * The cyclic corner elements are stored in each row's unused slot:
 * a[0] = beta (top-right corner, row 0's unused sub-diagonal)
 * c[n-1] = alpha (bottom-left corner, row n-1's unused super-diagonal)
 *
 * @param params  PCR double-buffer workspace
 * @param a, b, c, rhs  Input tridiagonal coefficients (length >= n)
 * @param out     Output array (length >= n), receives final result
 */
template <typename T>
__device__ void cyclic_tridiag_blockwide_pcr(tridiag_pcr_scratch<T> &params,
                                             const T *a, const T *b, const T *c,
                                             const T *rhs, T *out) {
  int n = params.length;

  // Convention: each row's corner is in that row's unused slot
  T beta = a[0];      // top-right corner (stored in row 0's unused a)
  T alpha = c[n - 1]; // bottom-left corner (stored in row n-1's unused c)

  // Sherman-Morrison vectors: mu (u) and nu (v)
  T gamma = -b[0];
  T mu_start = gamma;
  T mu_end = alpha;
  T nu_start = T(1);
  T nu_end = beta / gamma;

  // --- Solve 1: A'y = rhs ---
  // Copy inputs into buf0 with modified diagonal (corners removed)
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    params.buf0_a[i] = a[i];
    params.buf0_b[i] = b[i];
    params.buf0_c[i] = c[i];
    params.buf0_rhs[i] = rhs[i];
    if (i == 0) {
      params.buf0_a[i] = T(0);
      params.buf0_b[i] = b[i] - gamma; // b0' = b0 - gamma = 2*b0
    } else if (i == n - 1) {
      params.buf0_c[i] = T(0);
      params.buf0_b[i] =
          b[i] - mu_end * nu_end; // b_{n-1}' = b_{n-1} - alpha*beta/gamma
    }
  }
  __syncthreads();

  // Solve 1: write y[i] = rFinal[i] / bFinal[i] into out
  T *rFinal, *bFinal;
  pcr_reduce_blockwide<T>(params.buf0_a, params.buf0_b, params.buf0_c,
                          params.buf0_rhs, params.buf1_a, params.buf1_b,
                          params.buf1_c, params.buf1_rhs, n, rFinal, bFinal);

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = rFinal[i] / bFinal[i];
  }
  __syncthreads();

  // --- Solve 2: A'z = mu ---
  // Same A' (need to reload), RHS is the mu vector: (gamma, 0, ..., 0, alpha)
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    params.buf0_a[i] = a[i];
    params.buf0_b[i] = b[i];
    params.buf0_c[i] = c[i];
    params.buf0_rhs[i] = T(0);
    if (i == 0) {
      params.buf0_a[i] = T(0);
      params.buf0_b[i] = b[i] - gamma;
      params.buf0_rhs[i] = mu_start;
    } else if (i == n - 1) {
      params.buf0_c[i] = T(0);
      params.buf0_b[i] = b[i] - mu_end * nu_end;
      params.buf0_rhs[i] = mu_end;
    }
  }
  __syncthreads();

  // Solve 2: keep rFinal/bFinal pointers, don't copy to a separate array
  pcr_reduce_blockwide<T>(params.buf0_a, params.buf0_b, params.buf0_c,
                          params.buf0_rhs, params.buf1_a, params.buf1_b,
                          params.buf1_c, params.buf1_rhs, n, rFinal, bFinal);

  // --- Combine: x = y - (v^T y / (1 + v^T z)) * z ---
  // z[i] = rFinal[i] / bFinal[i] lives in the double buffers
  // v^T y = y[0] + beta/gamma * y[n-1]  (y is in out)
  // v^T z = z[0] + beta/gamma * z[n-1]  (z is in buffers)
  // All threads redundantly compute mult (cheap: 6 loads + ~8 FLOPs)
  T vTy = out[0] * nu_start + out[n - 1] * nu_end;
  T z0 = rFinal[0] / bFinal[0];
  T zn = rFinal[n - 1] / bFinal[n - 1];
  T vTz = z0 * nu_start + zn * nu_end;
  T mult = vTy / (T(1) + vTz);

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] -= mult * rFinal[i] / bFinal[i];
  }
  __syncthreads();
}

/**
 * @brief Cyclic tridiagonal solver kernel (global memory variant).
 *
 * The cyclic corner elements are packed into the unused slots of a[] and c[]:
 *   a[0]   = beta  (top-right corner)
 *   c[n-1] = alpha (bottom-left corner)
 *
 * @example For a 4x4 cyclic system:
 *
 *   | b0  c0   0   β |       a[] = { β,  a1, a2, a3 }
 *   | a1  b1  c1   0 |       b[] = { b0, b1, b2, b3 }
 *   |  0  a2  b2  c2 |       c[] = { c0, c1, c2, α  }
 *   |  α   0  a3  b3 |
 */
template <typename T>
__global__ void cyclic_tridiag_blockwise_pcr_kernel(
    const T *a, const T *b, const T *c, const T *rhs, T *u, T *buf0_a,
    T *buf0_b, T *buf0_c, T *buf0_rhs, T *buf1_a, T *buf1_b, T *buf1_c,
    T *buf1_rhs, const size_t *num_elements_in_row,
    const int stride_elements_per_row, const int num_rows) {

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    tridiag_pcr_scratch<T> params = {
        &buf0_a[off],   &buf0_b[off],   &buf0_c[off],
        &buf0_rhs[off], &buf1_a[off],   &buf1_b[off],
        &buf1_c[off],   &buf1_rhs[off], num_elements_in_row[blk]};
    cyclic_tridiag_blockwide_pcr<T>(params, &a[off], &b[off], &c[off],
                                    &rhs[off], &u[off]);
  }
}

} // end namespace cutridiag

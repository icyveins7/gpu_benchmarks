#pragma once

#include "sharedmem.cuh"

/**
 * @brief Device-wide kernel to compute the sum and the squared sum of an input.
 * This can be used to compute the mean and standard deviation in later kernels.
 *
 * @tparam T Input type
 * @tparam U Output type; all inputs are cast to this first before accumulation
 * @param useExplicitSharedMemory Highly recommended to leave as true, as it has
 * been measured to be faster.
 * @param in Input array
 * @param length Length of input
 * @param d_sum Global memory sum, assumed to be zeroed
 * @param d_sumSq Global memory squared sum, assumed to be zeroed
 */
template <typename T, typename U, bool useExplicitSharedMemory = true>
__global__ void device_sum_and_sumSq_kernel(const T *in, size_t length,
                                            U *d_sum, U *d_sumSq) {

  U x = 0;
  U xsq = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length;
       i += gridDim.x * blockDim.x) {
    // Iterate over all the data and collect both x and x^2
    U val = static_cast<U>(in[i]);
    x += val;
    xsq += val * val;
  }
  // Atomically update the sum and sumSq
  if constexpr (useExplicitSharedMemory) {
    SharedMemory<U> smem;
    U *s_workspace = smem.getPointer();
    // Use workspace to reduce first
    s_workspace[threadIdx.x] = x;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        s_workspace[threadIdx.x] += s_workspace[threadIdx.x + s];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(d_sum, s_workspace[0]);
    }

    // Similar reduction for the xsq
    s_workspace[threadIdx.x] = xsq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        s_workspace[threadIdx.x] += s_workspace[threadIdx.x + s];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(d_sumSq, s_workspace[0]);
    }
  }
  // Try to let the compiler optimize the atomics? NOTE: don't use this
  else {
    atomicAdd(d_sum, x);
    atomicAdd(d_sumSq, xsq);
  }
}

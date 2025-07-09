#pragma once

#include "atomic_extensions.cuh"
#include "sharedmem.cuh"

#include <cuda/std/limits>

/**
 * @brief Device-wide kernel to compute the sum and the sum of squares of an
 * input. This can be used to compute the mean and standard deviation in later
 * kernels.
 *
 * @tparam T Input type
 * @tparam U Output type; all inputs are cast to this first before accumulation
 * @param useExplicitSharedMemory Highly recommended to leave as false, as it
 * has been measured to be faster (using warp-level aggregation).
 * @param in Input array
 * @param length Length of input
 * @param d_sum Global memory sum, assumed to be zeroed
 * @param d_sumSq Global memory squared sum, assumed to be zeroed
 */
template <typename T, typename U, bool useExplicitSharedMemory = false>
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
  } else {
    // Do warp-level aggregation with atomics
    atomicAggIncSum(d_sum, x);
    atomicAggIncSum(d_sumSq, xsq);
    // TODO: this may not necessarily work properly if data is not a multiple of
    // 32
  }

  // NOTE: previously we tested a plain atomicAdd to see if CUDA 12.6 nvcc would
  // automatically optimize warp aggregation well. but this performed worse than
  // shared memory reduction.
}

/**
 * @brief Calculates the sum and sum of squares of individual equal-length
 * sections of an input, contingent on their validity (marked by the
 * ignoredValue, like a NaN).
 *
 * |--section 0--|--section 1--|--......
 *    │               └summed into d_sum[1], d_sumSq[1]
 *    └summed into d_sum[0], d_sumSq[0]
 *
 * NOTE: this kernel assumes a sufficient grid dimensions to cover the data.
 * 1) You should ensure that gridDim.y is equal to the number of sections.
 * 2) You should ensure that gridDim.x * blockDim.x >= maxSectionSize
 *
 * @tparam Tinput Data type of input
 * @tparam Toutput Data type of output sum and sum of squares
 * @param in Input array (numSections x maxSectionSize)
 * @param d_sum Output sums (per section)
 * @param d_sumSq Output sums of squares (per section)
 * @param maxSectionSize Section length (including invalid elements)
 * @param numSections Number of sections
 * @param validSectionSizes Output accumulator counter, excluding the invalid
 * elements (per section)
 * @param ignoredValue Value to ignore per element
 */
template <typename Tinput, typename Toutput, typename Tsize = int>
__global__ void device_sectioned_sum_and_sumSq_kernel(
    const Tinput *in, Toutput *d_sum, Toutput *d_sumSq, Tsize maxSectionSize,
    Tsize numSections, Tsize *validSectionSizes = nullptr,
    Tinput ignoredValue = cuda::std::numeric_limits<Tinput>::max()) {
  // Section index for the thread, assume the grid has accounted for this
  // NOTE: this is important, as we must have every warp contribute to the same
  // sectionIdx i.e. the grouping is such that blockIdx.x * blockDim.x spans the
  // sectionSize
  Tsize sectionIdx = blockIdx.y;
  if (sectionIdx >= numSections)
    return;

  // Input index for the thread
  Tsize idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Read inputs contingent on available data
  Tinput val;
  Toutput valOut, valSq;
  if (idx >= maxSectionSize) {
    valOut = 0;
    valSq = 0;
  } else {
    // No grid-stride, assume grid covers input data sufficiently
    val = in[sectionIdx * maxSectionSize + idx];
    // The value is set to 0 (nothing added) if it's to be ignored
    valOut = val == ignoredValue ? 0 : static_cast<Toutput>(val);
    valSq = valOut * valOut;
  }
  // Doing the above ensures that every thread in every warp has a valid value

  // Atomic warp-level aggregation
  atomicAggIncSum(&d_sum[sectionIdx], valOut);
  atomicAggIncSum(&d_sumSq[sectionIdx], valSq);

  // Increment our counter as well, if provided
  if (validSectionSizes != nullptr && val != ignoredValue &&
      idx < maxSectionSize)
    atomicAggInc(&validSectionSizes[sectionIdx]);
}

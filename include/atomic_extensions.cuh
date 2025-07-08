#pragma once
/* This header will contain kernel functions related to threshold checking of
 * data. Since this is usually just a simple check, it is probably not a good
 * idea to make this a __global__ kernel, but do this operation as part of
 * another kernel.
 *
 * Good references:
 * https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 * https://developer.nvidia.com/blog/cooperative-groups/
 */

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T> __device__ void atomicAggIncSum(T *sum, T val) {
  cg::coalesced_group g = cg::coalesced_threads();
  for (int i = g.size() / 2; i > 0; i /= 2) {
    val += g.shfl_down(val, i);
  }
  if (g.thread_rank() == 0)
    atomicAdd(sum, val);
}

/**
 * @brief Checks if an element is greater than the threshold, and atomically
 * increases the global counter. This is the most naive method, and the least
 * performant in the case where many elements exceed the threshold.
 *
 * The atomics' pressure can be reduced by using grid-stride techniques i.e. use
 * a smaller grid so each thread loops through multiple elements, and
 * accumulates the counter before the atomic write to global memory, but this
 * would require more custom code.
 *
 * NOTE: do not use this if you are intending to use a local thread variable for
 * the counter! Atomic overhead will be incurred!
 *
 * @tparam T Type of data.
 * @param count Counter. Could be in global/shared memory (do not use local!).
 * @param val Value to be compared
 * @param threshold Threshold to compare against
 * @return Returns true if the value is greater than the threshold,
 *         for internal use
 */
template <typename T, typename U = unsigned int>
__device__ bool exceedsThreshold(U *count, const T val, const T threshold) {
  bool exceeded = val > threshold;
  if (exceeded)
    atomicAdd(count, 1);
  return exceeded;
}

/**
 * @brief Warp-aggregated atomic add of a counter. This returns an
 * atomically-incremented index, corresponding to the counter, which can be used
 * to write other data (e.g. in filtering tasks where array elements are copied
 * conditionally depending on a predicate).
 *
 * TODO: nvcc is supposed to automatically do this past CUDA 9? It may actually
 * be more costly if we manually do it ourselves? See:
 * https://forums.developer.nvidia.com/t/cuda-pro-tip-optimized-filtering-with-warp-aggregated-atomics/148425
 *
 * @example
 *   if (value > threshold)
 *      output[atomicAggInc(counter)] = value;
 *
 * @tparam T Data type of the counter, corresponding to atomicAdd types
 * @param counter Counter to increment, can be in global or shared memory
 * @return The index to write to, per activated thread
 */
template <typename T> __device__ int atomicAggInc(T *counter) {
  // Gets all the threads that activated this in the warp
  auto g = cg::coalesced_threads();

  T warp_res;
  // Get your lowest active thread (leader thread)
  if (g.thread_rank() == 0)
    // Atomically increment by the number of threads active in this warp
    warp_res = atomicAdd(counter, g.size());

  // Return 'index' for each thread: warp_res is the leader thread's index and
  // the thread_rank will handle the rest in the warp
  // NOTE: the atomicAdd returned the value before it was incremented
  // TODO: check if this has some overhead (does compiler optimize away if
  // unused?)
  return g.shfl(warp_res, 0) + g.thread_rank();
}

/**
 * @brief Implements atomicMin for floats, since there isn't one provided by
 * CUDA. Taken from
 * https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda.
 * NOTE: this does not handle NaNs or inf comparisons correctly
 *
 * @param addr Address being atomically compared
 * @param value Value being atomically compared
 * @return Old value at address
 */
__device__ __forceinline__ float atomicMinFloat(float *addr, float value) {
  float old;
  // NOTE: this is doable since reinterpreting bits of a float retains ordering
  // as an integer, assuming both are positive (first bit is 0 i.e. sign bit)
  // the ternary below handles cases where one or the other is negative
  old = !signbit(value)
            ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMax((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

/**
 * @brief Implements atomicMax for floats, since there isn't one provided by
 * CUDA. Taken from
 * https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda.
 * NOTE: this does not handle NaNs or inf comparisons correctly
 *
 * @param addr Address being atomically compared
 * @param value Value being atomically compared
 * @return Old value at address
 */
__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  // NOTE: this is doable since reinterpreting bits of a float retains ordering
  // as an integer, assuming both are positive (first bit is 0 i.e. sign bit)
  // the ternary below handles cases where one or the other is negative
  old = !signbit(value)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

__device__ __forceinline__ double atomicMinDouble(double *addr, double value) {
  // The syntax for this is slightly more verbose since there are no direct type
  // cast intrinsics for unsigned long long to/from double, so we make it
  // consistent by doing the casting manually

  // Do min as long long int
  if (!signbit(value)) {
    long long int oldCast =
        atomicMin((long long int *)addr, *(long long int *)(&value));
    return *((double *)&oldCast);
  }
  // Do max as unsigned long long int
  else {
    unsigned long long int oldCast = atomicMax(
        (unsigned long long int *)addr, *(unsigned long long int *)(&value));
    return *((double *)&oldCast);
  }
}

__device__ __forceinline__ double atomicMaxDouble(double *addr, double value) {
  // The syntax for this is slightly more verbose since there are no direct type
  // cast intrinsics for unsigned long long to/from double, so we make it
  // consistent by doing the casting manually

  // Do max as long long int
  if (!signbit(value)) {
    long long int oldCast =
        atomicMax((long long int *)addr, *(long long int *)(&value));
    return *((double *)&oldCast);
  }
  // Do min as unsigned long long int
  else {
    unsigned long long int oldCast = atomicMin(
        (unsigned long long int *)addr, *(unsigned long long int *)(&value));
    return *((double *)&oldCast);
  }
}

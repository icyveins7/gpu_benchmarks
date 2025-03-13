#include "sharedmem.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Directly copied from https://developer.nvidia.com/blog/cooperative-groups/
template <typename T>
__device__ T reduce_sum(cg::thread_group g, T *temp, T val) {
  int lane = g.thread_rank();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = val;
    g.sync(); // wait for all threads to store
    if (lane < i)
      val += temp[lane + i];
    g.sync(); // wait for all threads to load
  }
  return val; // note: only thread 0 will return full sum
}

/**
 * @brief Naive block-to-output implementation of pattern matching kernel,
 * computing least squares residuals. Each block works on a single pattern and a
 * single input vector.
 *
 * @tparam T Type of input and pattern
 * @tparma U Type of metric calculations
 * @param d_inputs Array of input vectors, stacked contiguously
 * @param numInputs Number of array input vectors
 * @param d_patterns Array of pattern vectors, stacked contiguously
 * @param numPatterns Number of array pattern vectors
 * @param length Length of each input/pattern vector
 * @param d_metric Output array
 */
template <typename T, typename U = T>
__global__ void
naivePatternMatchKernel_LS(const T *d_inputs, // numInputs * length
                           const unsigned int numInputs,
                           const T *d_patterns, // numPatterns * length
                           const unsigned int numPatterns,
                           const unsigned int length,
                           U *d_metric // numInputs * numPatterns
) {
  // Retrieve the input and pattern that this block is working on
  const unsigned int pIdx = blockIdx.x;
  const unsigned int iIdx = blockIdx.y;

  // Exit if this block has no work
  if (pIdx >= numPatterns || iIdx >= numInputs)
    return;

  // Retrieve shared memory
  SharedMemory<U> smem;
  U *s_workspace = smem.getPointer(); // assume 1 for each thread

  // Otherwise get a pointer to the start of both vectors
  const T *b_input = d_inputs[iIdx * length];
  const T *b_pattern = d_patterns[pIdx * length];

  // Perform the least squares residual calculation on each vector
  U diff;
  U metric = 0;
  for (int t = threadIdx.x; t < length; t += blockDim.x) {
    diff = b_input[t] - b_pattern[t];
    // Accumulated into local thread workspace first
    metric += diff * diff;
  }

  // Block reduce to a sum
  __syncthreads();
  auto g = cg::this_thread_block();
  U sum = reduce_sum(g, s_workspace, metric);

  if (g.thread_rank() == 0)
    d_metric = sum; // no atomic needed here since 1 block writes 1 val
}

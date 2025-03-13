#pragma once

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
  const T *b_input = &d_inputs[iIdx * length];
  const T *b_pattern = &d_patterns[pIdx * length];

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
    d_metric[iIdx * numPatterns + pIdx] =
        sum; // no atomic needed here since 1 block writes 1 val
}

template <unsigned int TILE_ROWS, unsigned int TILE_COLS, typename T,
          typename U = T>
__global__ void
tiledPatternMatchKernel_LS(const T *d_inputs, // numInputs * length
                           const unsigned int numInputs,
                           const T *d_patterns, // numPatterns * length
                           const unsigned int numPatterns,
                           const unsigned int length,
                           U *d_minVal,           // numInputs
                           unsigned int *d_minIdx // numInputs
) {
  // Evaluate the tile that this block is working on
  const unsigned int pIdx0 = blockIdx.x * TILE_COLS;
  const unsigned int iIdx0 = blockIdx.y * TILE_ROWS;

  // Store all required input and pattern vectors in shared memory
  SharedMemory<T> smem;
  T *s_inputs = smem.getPointer();
  T *s_patterns = &s_inputs[TILE_ROWS * length];
  U *s_workspace = (U *)&s_patterns[TILE_COLS * length]; // 1 for each thread
  U *s_metric = (U *)&s_workspace[blockDim.x];           // tile dimensions
  unsigned int *s_minIdx =
      (unsigned int *)&s_metric[TILE_ROWS * TILE_COLS]; // 1 for each tile input
                                                        // i.e. TILE_ROWS

  for (unsigned int iIdx = 0; iIdx < TILE_COLS; iIdx++) {
    // Read only if valid
    if (iIdx0 + iIdx < numInputs) {
      for (int t = threadIdx.x; t < length; t += blockDim.x) {
        s_inputs[iIdx * length + t] = d_inputs[(iIdx0 + iIdx) * length + t];
      }
    }
  }
  for (unsigned int pIdx = 0; pIdx < TILE_COLS; pIdx++) {
    // Read only if valid
    if (pIdx0 + pIdx < numPatterns) {
      for (int t = threadIdx.x; t < length; t += blockDim.x) {
        s_patterns[pIdx * length + t] = d_patterns[(pIdx0 + pIdx) * length + t];
      }
    }
  }
  __syncthreads();

  // Now operate on the tile
  U diff, metric;
  for (unsigned int iIdx = 0; iIdx < TILE_ROWS; iIdx++) {
    for (unsigned int pIdx = 0; pIdx < TILE_COLS; pIdx++) {
      metric = 0;
      for (int t = threadIdx.x; t < length; t += blockDim.x) {
        diff = s_inputs[iIdx * length + t] - s_patterns[pIdx * length + t];
        metric += diff * diff;
      }

      // Block reduce to a sum
      __syncthreads();
      auto g = cg::this_thread_block();
      U sum = reduce_sum(g, s_workspace, metric);

      // Store the sum into shared memory output
      if (g.thread_rank() == 0)
        s_metric[iIdx * TILE_COLS + pIdx] = sum;
    }
  }

  __syncthreads();

  // Finally we compare the metrics internally to get the minimum
  // Assumption here is that blockDim.x <= TILE_ROWS i.e.
  // we can fully utilise the block by just scanning along each row with 1
  // thread
  for (int t = threadIdx.x; t < TILE_ROWS; t += blockDim.x) {
    // Initialise to the first index
    unsigned int minIdx = 0;
    U minVal = s_metric[t * TILE_COLS + 0];
    // NOTE: this way of iterating will have bank conflicts if TILE_COLS is a
    // multiple of 32
    for (int i = 0; i < TILE_COLS; i++) {
      if (s_metric[t * TILE_COLS + i] < minVal) {
        minVal = s_metric[t * TILE_COLS + i];
        minIdx = i;
      }
    }

    // We now update the minIdx of the tile to its global value
    minIdx += pIdx0;
    // And then atomically update the global values
    // TODO:
  }
}

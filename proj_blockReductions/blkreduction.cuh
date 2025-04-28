#pragma once

#include "cub/cub.cuh"
#include "sharedmem.cuh"
#include <stdint.h>

static_assert(true); // dummy assert just for clangd LSP to stop complaining

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

/**
 * @brief Max and argmax reduction within a block. This does not currently
 * (24/4/2025) seem to have a simple cub implementation. The max and argmax are
 * in the index 0 element of each array at the end, as usual. The argmax prefers
 * the earliest index in the case of equality.
 *
 * NOTE: This performs a syncthreads() for you at the start!
 *
 * @tparam T Type of input
 * @param sdata Input data in shared memory, assumed to have length blockDim.x
 * @param sidx Associated indices of the input data in shared memory, same
 * length
 */
template <typename T>
__device__ void blockReduceMaxAndArgMax(T *sdata, unsigned int *sidx) {
  // We won't assume the user will sync before this.
  __syncthreads();

  // Reduce inside shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x + s] >= sdata[threadIdx.x]) {
        // Prefer the earlier index if equal
        sidx[threadIdx.x] = sdata[threadIdx.x] == sdata[threadIdx.x + s]
                                ? min(sidx[threadIdx.x], sidx[threadIdx.x + s])
                                : sidx[threadIdx.x + s];
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
}

/**
 * @brief Block-wise argmax while loading into registers. This is usually the
 * preparation step before the block-wise reduction via shared memory. The
 * argmax prefers the earliest index in the case of equality.
 *
 * @tparam T Type of input
 * @param blk_in Input array for the block
 * @param inLenPerBlock Length of input array for the block
 * @param r_argmax Output (Register) argmax per thread
 * @param r_max Output (Register) max per thread
 */
template <typename T>
__device__ void blockArgmaxWhileLoading(const T *blk_in,
                                        const int inLenPerBlock,
                                        unsigned int &r_argmax, T &r_max) {
  for (int t = threadIdx.x; t < inLenPerBlock; t += blockDim.x) {
    T candidate = blk_in[t];
    // On first iteration simply take the value
    if (t == threadIdx.x) {
      r_max = candidate;
      r_argmax = t;
    }

    // On subsequent iterations compare
    if (candidate >= r_max) {
      // Prefer the earlier index if equal
      r_argmax = candidate == r_max ? min(r_argmax, t) : t;
      r_max = blk_in[t];
    }
  }
}

/**
 * @brief Simple example of how to apply the device block reduction argmax
 * function.
 *
 * @tparam T Type of input
 * @param in Contiguous input arrays, each block works on one array
 * @param inLenPerBlock Length per block's input array
 * @param maxPerBlock Maximum value per input array/block
 * @param argMaxPerBlock Argmax per input array/block
 * @param debugMaxPerBlock For debugging: shared memory initialization of max
 * @param debugArgMaxPerBlock For debugging: shared memory initialization of
 * argmax
 */
template <typename T>
__global__ void
simpleBlockMaxAndArgMaxKernel(const T *in, const int inLenPerBlock,
                              T *maxPerBlock, unsigned int *argMaxPerBlock,
                              T *debugMaxPerBlock = nullptr,
                              unsigned int *debugArgMaxPerBlock = nullptr) {
  // Get the input for the block
  const T *blk_in = &in[blockIdx.x * inLenPerBlock];

  // Get some shared mem for 1 block
  SharedMemory<T> smem;
  T *s_in = smem.getPointer();
  unsigned int *s_idx = (unsigned int *)&s_in[blockDim.x];

  // Compare while loading
  T r_in; // compare and load in registers first
  unsigned int r_idx;
  blockArgmaxWhileLoading(blk_in, inLenPerBlock, r_idx, r_in);
  // Update our shared memory values
  s_in[threadIdx.x] = r_in;
  s_idx[threadIdx.x] = r_idx;

  // Debug shared memory states
  if (debugMaxPerBlock != nullptr) {
    debugMaxPerBlock[blockIdx.x * blockDim.x + threadIdx.x] = s_in[threadIdx.x];
  }
  if (debugArgMaxPerBlock != nullptr) {
    debugArgMaxPerBlock[blockIdx.x * blockDim.x + threadIdx.x] =
        s_idx[threadIdx.x];
  }

  // Reduce in shared mem
  blockReduceMaxAndArgMax(s_in, s_idx);

  if (threadIdx.x == 0) {
    maxPerBlock[blockIdx.x] = s_in[0];
    argMaxPerBlock[blockIdx.x] = s_idx[0];
  }
}

/**
 * @brief Simple example of how to use cub to do Argmax
 *
 * @tparam T Type of data
 * @param in Input arrays
 * @param inLenPerBlock Length of each individual input array
 * @param maxPerBlock Max value for each array
 * @param argMaxPerBlock Argmax for each array
 */
template <typename T, int BlockSize>
__global__ void cubArgmax(const T *in, const int inLenPerBlock, T *maxPerBlock,
                          unsigned int *argMaxPerBlock) {
  // See cub/util_type.cuh for the cub::KeyValuePair struct
  using BlkRed =
      cub::BlockReduce<cub::KeyValuePair<unsigned int, T>, BlockSize>;
  // Essentially allocates shared memory block of size BlockSize
  __shared__ typename BlkRed::TempStorage temp_storage;

  // Get the input for the block
  const T *blk_in = &in[blockIdx.x * inLenPerBlock];

  // Compare while loading
  T r_in; // compare and load in registers first
  unsigned int r_idx;
  blockArgmaxWhileLoading(blk_in, inLenPerBlock, r_idx, r_in);

  // Create the key value pair to do block-wise reduction
  cub::KeyValuePair<unsigned int, T> toReduce(r_idx, r_in);
  // Invoke cub's reduction directly (apparently no need to syncthreads before
  // hand)
  cub::KeyValuePair<unsigned int, T> aggregate =
      BlkRed(temp_storage).Reduce(toReduce, cub::ArgMax());
  // Also no need to syncthreads after

  if (threadIdx.x == 0) {
    maxPerBlock[blockIdx.x] = static_cast<T>(aggregate.value);
    argMaxPerBlock[blockIdx.x] = static_cast<unsigned int>(aggregate.key);
  }
}

#pragma GCC diagnostic pop

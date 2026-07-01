#pragma once

/*
 * See with scan_short_words.cu.
 * This does appear to be slightly faster 1.067ms vs 1.224ms, for 16384 x 16384.
 * One block per row, 256 threads per block.
 *
 * Serves as a good guide for future kernels that can assure full utilization
 * while assigning one block per row.
 */

#include "containers/image.cuh"
#include "cub/cub.cuh"

template <typename T, typename Tpacked, int NUM_THREADS,
          typename ScanOp = cuda::maximum<T>>
__global__ void packedwords_rowwise_inclusive_scan_kernel(
    const containers::Image<const T> input, containers::Image<T> output,
    ScanOp op = cuda::maximum<T>{}) {
  constexpr int PACKSIZE = sizeof(Tpacked) / sizeof(T);

  // Each block handles one row
  if (blockIdx.x >= input.height)
    return;

  using BlockScan = cub::BlockScan<T, NUM_THREADS>;
  __shared__ typename BlockScan::TempStorage tempStorage;
  __shared__ Tpacked s_data[NUM_THREADS]; // +1 in case of incomplete word
  T* s_data_small = (T*)s_data;

  int offset = 0;
  T aggregate = 0; // initial value aggregate
  while (offset < input.width) {
    // Define number of actual values (unpacked)
    int numThisIter = min(NUM_THREADS * PACKSIZE, input.width - offset);

    Tpacked* inputPackedOffset = (Tpacked*)&input.at(blockIdx.x, offset);
    Tpacked* outputPackedOffset = (Tpacked*)&output.at(blockIdx.x, offset);

    // Load
    if (numThisIter % PACKSIZE == 0) {
      // Load full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        s_data[t] = inputPackedOffset[t];
      }
    } else {
      // Load normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        s_data_small[t] = input.at(blockIdx.x, offset + t);
      }
    }
    // Handle first thread inclusion (from last iteration)
    if (offset != 0 && threadIdx.x == 0) {
      s_data_small[0] = op(aggregate, s_data_small[0]);
    }
    __syncthreads();

    // Perform inclusive scan repeatedly as unpacked values
    // NOTE: the aggregate on the final scan is not correct if it the iteration
    // does not fully utilize the entire shared memory length, but this isn't
    // important anyway since the aggregate would no longer be needed after.
    int numRepeatScans =
        numThisIter / NUM_THREADS + (numThisIter % NUM_THREADS == 0 ? 0 : 1);
    for (int i = 0; i < numRepeatScans; i++) {
      BlockScan(tempStorage)
          .InclusiveScan(s_data_small[i * NUM_THREADS + threadIdx.x],
                         s_data_small[i * NUM_THREADS + threadIdx.x], op,
                         aggregate);
      // fix next scan's first value using aggregate
      if (i < numRepeatScans - 1 && threadIdx.x == 0) {
        s_data_small[(i + 1) * NUM_THREADS] =
            op(aggregate, s_data_small[(i + 1) * NUM_THREADS]);
      }
      __syncthreads(); // for reuse of tempStorage
    }

    // Write out
    if (numThisIter % PACKSIZE == 0) {
      // Store full words, one packed word per thread
      for (int t = threadIdx.x; t < numThisIter / PACKSIZE; t += blockDim.x) {
        outputPackedOffset[t] = s_data[t];
      }
    } else {
      // Store normally as unpacked values
      for (int t = threadIdx.x; t < numThisIter; t += blockDim.x) {
        output.at(blockIdx.x, offset + t) = s_data_small[t];
      }
    }

    // sync before next iteration, also needed to reuse BlockScan
    offset += numThisIter;
    __syncthreads();
  }
}

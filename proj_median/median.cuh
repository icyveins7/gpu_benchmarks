#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda/std/limits>

template <typename T, int NUM_THREADS_PER_BLK, int ELEM_PER_THREAD>
__global__ void blockwise_median_kernel(
    const T *inputRows,      // the input data, numRows x numColumns
    const int numRows,       // number of rows
    const int numColumns,    // number of columns
    const int *inputLengths, // the number of actually used elements in each row
    T *medians) {
  // TODO: move this into docstring above when ready
  // - Expect that invalid data is filled with std::numeric_limits<T>::max()
  // This will allow us to read the entire row, and sort as a static length,
  // then ignore the ending length i.e get the validLength/2 as median
  // - The numColumns represents the input data's dimensions; this should be
  // smaller than or equal to NUM_THREADS * ELEM_PER_THREAD. It does not REQUIRE
  // it to be EQUAL.

  // Define the sort length
  constexpr int SORT_LENGTH = NUM_THREADS_PER_BLK * ELEM_PER_THREAD;

  // Each block processes one row
  int row = blockIdx.x;
  // Exit if nothing to do
  if (row > numRows)
    return;

  // Check whether the sort length is valid (must be more than the input length
  // of this row) Otherwise write the max value for this type
  if (SORT_LENGTH < inputLengths[row] && threadIdx.x == 0) {
    medians[row] = cuda::std::numeric_limits<T>::max();
    return;
  }

  // Extract the row that will be processed
  const T *inputRow = &inputRows[row * numColumns];
  // Read the valid length for the row
  const int validLength = inputLengths[row];

  // Standard cub boilerplate, adapted from example_block_radix_sort.cu
  using BlockRadixSort =
      cub::BlockRadixSort<T, NUM_THREADS_PER_BLK, ELEM_PER_THREAD>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  // Per-thread tile items
  T items[ELEM_PER_THREAD];
  // Fill our per-thread tile items, globally coalesced
  // 1st loop fills index 0, 2nd loop fills index 1, etc
  // There should be a total of ELEM_PER_THREAD loops
  for (int t = threadIdx.x; t < NUM_THREADS_PER_BLK * ELEM_PER_THREAD;
       t += NUM_THREADS_PER_BLK) {
    // We write the valid elements
    if (t < validLength)
      items[t / NUM_THREADS_PER_BLK] = inputRow[t];
    // Fill the remainder with the max type value
    else
      items[t / NUM_THREADS_PER_BLK] = cuda::std::numeric_limits<T>::max();
  }

  // Now sort
  BlockRadixSort(temp_storage).Sort(items);

  // The median should be in the element corresponding to half valid length
  // We choose to ignore the even-length case and just use half valid length
  const int medianIdx = validLength / 2;
  const int threadIdxContainingMedian = medianIdx / ELEM_PER_THREAD;
  if (threadIdx.x == threadIdxContainingMedian)
    medians[row] = items[medianIdx % ELEM_PER_THREAD];
}

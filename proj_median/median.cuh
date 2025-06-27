#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda/std/limits>

/**
 * @brief Performs a row-wise median of a 2D input array, with one block
 * tackling each row. Each row may have a different occupied length, as
 * specified by a separate array.
 *
 * Fundamentally, this performs a cub::BlockRadixSort, where it sets the unused
 * elements to be numeric_limits<T>::max(). This allows us to extract the
 * 'correct' median by referencing the valid length.
 *
 * NOTE: median is defined to be a simple length/2, so even-length rows may not
 * be as expected, but it is more efficient to process this way.
 *
 * @tparam T Type of input data
 * @param inputRows The input data
 * @param numRows The number of input rows
 * @param numColumns The (max) number of input columns, not all may be used
 * @param inputLengths The number of actually used elements in each row
 * @param medians Output median values for each row
 */
template <typename T, int NUM_THREADS_PER_BLK, int ELEM_PER_THREAD>
__global__ void blockwise_median_kernel(
    const T *inputRows,      // the input data, numRows x numColumns
    const int numRows,       // number of rows
    const int numColumns,    // number of columns
    const int *inputLengths, // the number of actually used elements in each row
    T *medians               // numRows
) {
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

  // Special case: if the length is 0, we return without writing anything
  // Otherwise, due to how we fill the remainder elements, it will write
  // numeric_limits<T>::max() instead.
  if (inputLengths[row] == 0)
    return;

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

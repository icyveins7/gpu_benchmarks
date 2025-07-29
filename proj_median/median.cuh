#pragma once

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda/std/limits>

#include "atomic_extensions.cuh"
#include "sharedmem.cuh"

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
template <typename T, int NUM_THREADS_PER_BLK, int ELEM_PER_THREAD,
          bool allValidInFront = true>
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
  if (row >= numRows)
    return;

  // Read the valid input length for this row
  const int validLength = inputLengths[row];

  // Check whether the sort length is valid (must be more than the input length
  // of this row) Otherwise write the max value for this type
  if (SORT_LENGTH < validLength) {
    if (threadIdx.x == 0) {
      medians[row] = cuda::std::numeric_limits<T>::max();
    }
    return;
  }

  // Special case: if the length is 0, we return without writing anything
  // Otherwise, due to how we fill the remainder elements, it will write
  // numeric_limits<T>::max() instead.
  if (validLength == 0)
    return;

  // Extract the row that will be processed
  size_t rowOffset = row * numColumns; // may be very large, so use size_t here
  const T *inputRow = &inputRows[rowOffset];

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
    // Compile-time switch based on the type of data supplied
    // Here we assume that the first validLength elements of each row are used
    // E.g. O O O O O X X ...
    if constexpr (allValidInFront) {
      // we write the valid elements from the front
      if (t < validLength)
        items[t / NUM_THREADS_PER_BLK] = inputRow[t];
      // Fill the remainder with the max type value
      else
        items[t / NUM_THREADS_PER_BLK] = cuda::std::numeric_limits<T>::max();
    }
    // Otherwise, we cannot assume the first validLength elements are to be used
    // and we instead assume that the invalid elements have already been marked
    // as numeric_limits<T>::max(), so we read all the elements available
    // E.g. O O X O O X O X O ...
    else {
      if (t < numColumns)
        items[t / NUM_THREADS_PER_BLK] = inputRow[t];
      // Fill the remainder with the max type value
      else
        items[t / NUM_THREADS_PER_BLK] = cuda::std::numeric_limits<T>::max();
    }
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

template <typename T, int MAX_LENGTH> struct BlockQuickSelect {
  T sdata[MAX_LENGTH * 2];
  int scounters[2];

  __device__ void fill(int usedLength, const T *d_row) {
    for (int t = threadIdx.x; t < usedLength; t += blockDim.x) {
      if (t < MAX_LENGTH)
        this->sdata[t] = d_row[t];
    }
    __syncthreads();
  }

  __device__ T select(int length, int n) {
    // Entire data for the block starts in first half of sdata

    T *sdata1 = sdata;
    T *sdata2 = &sdata1[MAX_LENGTH];

    T pivot;
    int pivotIdx;
    int offset = 0;
    // Iterations should never exceed this anyway, just as a soft cap
    for (int i = 0; i < MAX_LENGTH; ++i) {
      // Get pivot, just use first element
      pivotIdx = i % length;
      pivot = sdata1[pivotIdx + offset];

      // Reset counters
      if (threadIdx.x == 0)
        scounters[0] = 0;
      if (threadIdx.x == 1)
        scounters[1] = 0;
      __syncthreads();

      // Loop over the side we want to recurse into
      for (int t = threadIdx.x; t < length; t += blockDim.x) {
        // Skip pivot
        if (t == pivotIdx)
          continue;

        // Read current element to compare
        // There is an additional offset because we may be reading from
        // the back loaded section of a previous iteration e.g.
        // LLLLXXXRRRRRRRR
        //        └start of the array for the right side
        T element = sdata1[t + offset];

        // Check <
        if (element < pivot) {
          // Front load to 2nd buffer
          int oIdx = atomicAggInc(&scounters[0]);
          sdata2[oIdx] = element;
        }
        // Check >
        else {
          // Back load to 2nd buffer
          int oIdx = MAX_LENGTH - atomicAggInc(&scounters[1]) - 1;
          sdata2[oIdx] = element;
        }
      }
      __syncthreads();

      // Which side do we recurse to?
      if (n == scounters[0]) {
        // e.g. looking for index 5, and there are 5
        // elements on left, then pivot is the selection
        break;
      } else if (n < scounters[0]) {
        // e.g. looking for index 5, and there are 6 elements on left i.e. 0-5
        // then element we want is on the left
        length = scounters[0];
        // no need to change 'n' since the 'n'th element in the whole is still
        // the 'n'th in the left
        // offset must now be 0
        offset = 0;
      } else {
        // e.g. looking for index 5, and there are 4 elements on left i.e. 0-3
        // then element we want is on the right
        length = scounters[1];
        // change the 'n' in the recursed shorter array on the right
        // in above example, 'n' would now be 0 [(0-3), pivot, target, ...)]
        n = n - scounters[0] - 1;
        // make sure our offset is now to the start of the right side
        offset = MAX_LENGTH - scounters[1];
      }
      __syncthreads();

      swapBuffers(&sdata1, &sdata2);
    } // end while loop

    // if we reach this without getting a value, use a helpful error value
    return pivot;
  } // end select() function

  __device__ void swapBuffers(T **sdata1, T **sdata2) {
    T *spare = *sdata2;
    *sdata2 = *sdata1;
    *sdata1 = spare;
  }
};

template <typename T, int MAX_LENGTH>
__global__ void blockwise_quickselect_kernel(const T *d_input, int numRows,
                                             int numCols, int *lengths,
                                             int *d_n, T *d_output) {
  // Get the row for this block
  int row = blockIdx.x;
  if (row >= numRows)
    return;
  const T *d_row = &d_input[row * numCols];
  // And the actual used length and selected element
  int usedLength = lengths[row];
  int n = d_n[row];

  // Call quickselect (now templated for max length with shared memory
  // allocation directly)
  __shared__ BlockQuickSelect<T, MAX_LENGTH> selector;
  selector.fill(usedLength, d_row);
  T nthElement = selector.select(usedLength, n);
  if (threadIdx.x == 0)
    d_output[row] = nthElement;
}

#pragma once

template <typename T = unsigned int> struct SliceBounds {
  T start;
  T end;
};

/**
 * @brief Selects (copies) 1D slices from an input array into a 2D output
 * array; the output array is effectively a 2D rectangular array where each
 * element contains oMaxLength elements.
 *
 * This is useful when we would like to have multiple slices and gather them
 * (unordered) for each output element (for example, to calculate the
 * mean/median). A slice is defined by the above struct SliceBounds, where start
 * and end index is defined.
 *
 * @tparam T Data type of input/output elements
 * @param d_input Input 2D array, iLength
 * @param iLength Input length, used for checking
 * @param d_output Output 2D array, oRows * oMaxLength
 * @param outputLengths Used output length for each element, oRows
 * @param oRows Output rows
 * @param oMaxLength Output maximum length
 * @param sliceIdx Slice indices for each output element, oRows * MAX_SLICES
 * @param numSlices Number of slices actually used (out of MAX_SLICES)
 * @param numSkipPerSlice Number of elements to skip in each slice (downsample
 * rate, optional)
 */
template <typename T, typename Tslice, int MAX_SLICES,
          typename Tskip = unsigned char>
__global__ void blockwise_select_1d_slices_kernel(
    const T *d_input,                      // iLength
    const unsigned int iLength,            // input dimensions
    T *d_output,                           // oRows * oMaxLength
    unsigned int *outputLengths,           // oRows
    const unsigned int oRows,              // output dimensions
    const unsigned int oMaxLength,         // see above
    const SliceBounds<Tslice> *sliceIdx,   // oRows * MAX_SLICES
    const unsigned int *numSlices,         // oRows
    const Tskip *numSkipPerSlice = nullptr // oRows
) {
  // Each block operates on one output element
  const unsigned int row = blockIdx.x;
  if (row >= oRows) {
    return;
  }

  // Read slice index for this output element
  const SliceBounds<Tslice> *blockSlices = &sliceIdx[row * MAX_SLICES];
  // Read number of slices
  const int numSlicesForThisBlock = numSlices[row];
  // Read (optional) skipping per slice for this row
  Tskip numSkipPerSliceForThisBlock =
      numSkipPerSlice ? numSkipPerSlice[row] : 1;

  // Iterate over slice bounds
  unsigned int lengthUsed = 0; // accumulate the length used
  for (int i = 0; i < numSlicesForThisBlock; ++i) {
    // Read next slice
    const SliceBounds slice = blockSlices[i];
    unsigned int length = slice.end - slice.start + 1;

    // Amend the length based on the skip rate
    // NOTE: this is equivalent to checking if the sliceIdx % numSkip == 0
    length = length / numSkipPerSliceForThisBlock +
             (length % numSkipPerSliceForThisBlock > 0 ? 1 : 0);
    // E.g. for length 21, skip 10 -> 0, 10, 20 are used

    // Check that the slice references a valid input
    // NOTE: we assume unsigned integer types, so no need to check below 0
    if (slice.start >= iLength || slice.end >= iLength) {
      continue; // ignore this slice and try the rest
    }

    // Copy the slice into the output
    for (unsigned int t = threadIdx.x; t < length; t += blockDim.x) {
      // Only write if our output has sufficient space for the slice
      if (lengthUsed + t < oMaxLength) {
        d_output[row * oMaxLength + lengthUsed + t] =
            d_input[slice.start + t * numSkipPerSliceForThisBlock];
      }
    } // end loop over current slice copy

    // Add the length used to accumulator
    // NOTE: since we accumulate on a per-thread basis there is no need to
    // syncthreads! each thread will know how much to go forward
    lengthUsed += length;
  } // end loop over slices

  // Write the final output length used
  if (threadIdx.x == 0)
    outputLengths[row] = lengthUsed < oMaxLength ? lengthUsed : oMaxLength;
};

// TODO: add kernel to perform multiple outputs per block.
// this would be useful especially if neighbouring outputs tend to have
// overlapping slices. the idea would be to find 'umbrella' slices, load them
// into shmem and then read from there for the various outputs, allowing for
// coalesced reads (the above kernel will suffer from non-coalesced reads,
// especially when downsampling is used)

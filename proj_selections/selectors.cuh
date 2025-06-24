#pragma once

template <typename T = unsigned int> struct SliceBounds {
  T start;
  T end;
};

/**
 * @brief Selects (copies) 1D slices from a 2D input array into a 3D output
 * array; the output array is effectively a 2D rectangular region (rows *
 * columns) where each element contains oMaxLength elements.
 *
 * This is useful when we would like to have multiple slices and gather them
 * (unordered) for each output element (for example, to calculate the
 * mean/median). A slice is defined by the above struct SliceBounds, where a row
 * and columnStart/End is defined.
 *
 * @tparam T Data type of input/output elements
 * @param d_input Input 2D array, iRows * iCols
 * @param iRows Input rows
 * @param iCols Input columns
 * @param d_output Output 3D array, oRows * oCols * oMaxLength
 * @param outputLengths Used output length for each element, same dimensions as
 * d_output
 * @param oRows Output rows
 * @param oCols Output columns
 * @param oMaxLength Output maximum length
 * @param sliceIdx Slice indices for each output element, oRows * oCols *
 * MAX_SLICES
 * @param numSlices Number of slices actually used (out of MAX_SLICES)
 */
template <typename T, typename Tslice, int MAX_SLICES>
__global__ void blockwise_select_1d_slices_kernel(
    const T *d_input,                    // iLength
    const unsigned int iLength,          // input dimensions
    T *d_output,                         // oRows * oMaxLength
    unsigned int *outputLengths,         // oRows
    const unsigned int oRows,            // output dimensions
    const unsigned int oMaxLength,       // see above
    const SliceBounds<Tslice> *sliceIdx, // oRows * MAX_SLICES
    const unsigned int *numSlices        // oRows
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

  // Iterate over slice bounds
  unsigned int lengthUsed = 0; // accumulate the length used
  for (int i = 0; i < numSlicesForThisBlock; ++i) {
    // Read next slice
    const SliceBounds slice = blockSlices[i];
    unsigned int length = slice.end - slice.start + 1;
    // Check that the slice references a valid input
    // NOTE: we assume unsigned integer types, so no need to check below 0
    if (slice.start >= iLength || slice.end >= iLength) {
      continue; // ignore this slice and try the rest
    }

    // Copy the slice into the output
    for (unsigned int t = threadIdx.x; t < length; t += blockDim.x) {
      // Only write if our output has sufficient space for the slice
      if (lengthUsed + t < oMaxLength) {
        d_output[row * oMaxLength + lengthUsed + t] = d_input[slice.start + t];
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

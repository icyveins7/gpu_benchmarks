#pragma once

struct SliceBounds {
  unsigned int row;
  unsigned int colStart;
  unsigned int colEnd;
};

template <typename T, int MAX_SLICES>
__global__ void blockwise_select_1d_slices_for_2d_kernel(
    const T *d_input,                                   // iRows * iCols
    const unsigned int iRows, const unsigned int iCols, // input dimensions
    T *d_output,                 // oRows * oCols * oMaxLength
    unsigned int *outputLengths, // oRows * oCols
    const unsigned int oRows, const unsigned int oCols, // output dimensions
    const unsigned int oMaxLength,                      // see above
    const SliceBounds *sliceIdx,  // oRows * oCols * MAX_SLICES
    const unsigned int *numSlices // oRows * oCols
) {
  // Each block operates on one output element
  const unsigned int row = blockIdx.y;
  const unsigned int col = blockIdx.x;
  if (row >= oRows || col >= oCols) {
    return;
  }

  // Read slice index for this output element
  const SliceBounds *blockSlices =
      &sliceIdx[row * oCols * MAX_SLICES + col * MAX_SLICES];
  // Read number of slices
  const int numSlicesForThisBlock = numSlices[row * oCols + col];

  // Iterate over slice bounds
  unsigned int lengthUsed = 0; // accumulate the length used
  for (int i = 0; i < numSlicesForThisBlock; ++i) {
    // Read next slice
    const SliceBounds slice = blockSlices[i];
    unsigned int length = slice.colEnd - slice.colStart + 1;
    // Check that the slice references a valid input
    if (slice.row >= iRows || slice.colStart >= iCols ||
        slice.colEnd >= iCols) {
      break;
    }

    // Copy the slice into the output
    for (unsigned int t = threadIdx.x; t < length; t += blockDim.x) {
      // Only write if our output has sufficient space for the slice
      if (lengthUsed + t < oMaxLength) {
        d_output[row * oCols * oMaxLength + col * oMaxLength + lengthUsed + t] =
            d_input[slice.row * iCols + slice.colStart + t];
      }
    } // end loop over current slice copy

    // Add the length used to accumulator
    // NOTE: since we accumulate on a per-thread basis there is no need to
    // syncthreads! each thread will know how much to go forward
    lengthUsed += length;
  } // end loop over slices

  // Write the final output length used
  if (threadIdx.x == 0)
    outputLengths[row * oCols + col] =
        lengthUsed < oMaxLength ? lengthUsed : oMaxLength;
};

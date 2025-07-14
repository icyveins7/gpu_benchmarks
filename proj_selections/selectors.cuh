#pragma once

#include <cuda/std/limits>

template <typename T = unsigned int> struct SliceBounds {
  T start;
  T end;
};

/**
 * @brief Internal function for blockwise_select_1d_slices_kernel.
 * Used when skipInSlice is true.
 */
template <typename T, typename Tslice, typename Tskip>
__device__ unsigned int blockwise_process_slices_skipInSlice(
    const int numSlicesForThisBlock, const SliceBounds<Tslice> *blockSlices,
    const Tskip *numSkip, const unsigned int row, const T *d_input,
    const unsigned int iLength, const unsigned int oMaxLength, T *d_output) {
  // Define the lengthUsed for the block, to return
  unsigned int lengthUsed = 0;
  // Read (optional) skipping per slice for this row
  Tskip numSkipPerSliceForThisBlock = numSkip ? numSkip[row] : 1;

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

  return lengthUsed;
}

/**
 * @brief Internal function for blockwise_select_1d_slices_kernel.
 * Used when skipInSlice is false.
 */
template <typename T, typename Tslice, typename Tskip>
__device__ unsigned int blockwise_process_slices_skipAfterSlices(
    const int numSlicesForThisBlock, const SliceBounds<Tslice> *blockSlices,
    const Tskip *numSkip, const unsigned int row, const T *d_input,
    const unsigned int iLength, const unsigned int oMaxLength, T *d_output) {
  // Define the raw (not downsampled) lengthUsed for the block, to return
  unsigned int rawLengthUsed = 0;
  // Read (optional) skipping per slice for this row
  Tskip numSkipForThisBlock = numSkip ? numSkip[row] : 1;

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
      // We only need to write the elements where the aggregated index is a
      // multiple of the downsample rate
      unsigned int aggInIdx = rawLengthUsed + t;
      unsigned int aggOutIdx = aggInIdx / numSkipForThisBlock;
      // Only write if our output has sufficient space for the slice
      if (aggInIdx % numSkipForThisBlock == 0 && aggOutIdx < oMaxLength) {
        // NOTE: this will be 'coalesced' on reads, but not on writes
        d_output[row * oMaxLength + aggOutIdx] = d_input[slice.start + t];
      }
    } // end loop over current slice copy

    // Add the length used to accumulator
    // NOTE: since we accumulate on a per-thread basis there is no need to
    // syncthreads! each thread will know how much to go forward
    rawLengthUsed += length;
  } // end loop over slices

  // At the end, we tweak the lengthUsed directly to account for the skip
  unsigned int lengthUsed = rawLengthUsed / numSkipForThisBlock +
                            (rawLengthUsed % numSkipForThisBlock > 0 ? 1 : 0);

  return lengthUsed;
}

/**
 * @brief Selects (copies) 1D slices from an input array into a 2D output
 * array; the output array is effectively a 2D rectangular array where each
 * element contains oMaxLength elements.
 *
 * This is useful when we would like to have multiple slices and gather them
 * (unordered) for each output element (for example, to calculate the
 * mean/median). A slice is defined by the above struct SliceBounds, where
 * start and end index is defined.
 *
 * @tparam T Data type of input/output elements
 * @tparam Tslice Data type of slice indices
 * @tparam MAX_SLICES Maximum number of slices
 * @tparam Tskip Data type of skip rate
 * @tparam skipInSlice Whether to skip elements in the slice (true) or skip
 * after aggregation (false)
 * @param d_input Input 2D array, iLength
 * @param iLength Input length, used for checking
 * @param d_output Output 2D array, oRows * oMaxLength
 * @param outputLengths Used output length for each element, oRows
 * @param oRows Output rows
 * @param oMaxLength Output maximum length
 * @param sliceIdx Slice indices for each output element, oRows * MAX_SLICES
 * @param numSlices Number of slices actually used (out of MAX_SLICES)
 * @param numSkip Number of elements to skip, based on template parameter
 * skipInSlice (downsample rate, optional)
 */
template <typename T, typename Tslice, int MAX_SLICES,
          typename Tskip = unsigned char, bool skipInSlice = false>
__global__ void blockwise_select_1d_slices_kernel(
    const T *d_input,                    // iLength
    const unsigned int iLength,          // input dimensions
    T *d_output,                         // oRows * oMaxLength
    unsigned int *outputLengths,         // oRows
    const unsigned int oRows,            // output dimensions
    const unsigned int oMaxLength,       // see above
    const SliceBounds<Tslice> *sliceIdx, // oRows * MAX_SLICES
    const unsigned int *numSlices,       // oRows
    const Tskip *numSkip = nullptr       // oRows
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
  // NOTE: we do not explicitly check for numSlicesForThisBlock > MAX_SLICES
  // since we assume it is prepared correctly

  unsigned int lengthUsed = 0;
  // Compile-time swap based on the skip method
  if constexpr (skipInSlice) {
    // Skip within each slice individually
    lengthUsed = blockwise_process_slices_skipInSlice<T, Tslice, Tskip>(
        numSlicesForThisBlock, blockSlices, numSkip, row, d_input, iLength,
        oMaxLength, d_output);
  } else {
    // Skip only after aggregating all slices
    lengthUsed = blockwise_process_slices_skipAfterSlices<T, Tslice, Tskip>(
        numSlicesForThisBlock, blockSlices, numSkip, row, d_input, iLength,
        oMaxLength, d_output);
  }

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

// ========================================================================
// ========================================================================
// ========================================================================

/**
 * @brief Downsamples from a rectangular ROI in an input array, conditioned on
 * both the values of the input as well as the values of a 2nd array, writing
 * invalid values when the validator says so.
 *
 * @details An example is included below, where a MaximumValidator can be used
 * to scan only the condition array for a maximum value, and returning an
 * invalid value otherwise.
 *
 * Effectively, this is an if-else condition like:
 *    if (condition < max)
 *      write downsampled value
 *    else
 *      write invalid value
 *
 * @tparam T Data type of arrays
 * @param input Input array, where data is downsampled
 * @param conditional Conditional input array, same dimensions as input
 * @param maxConditionVal Maximum value allowed in the conditional array
 * @param inputNumRows Number of rows of 'input'
 * @param inputNumCols Number of columns of 'input'
 * @param roiStartRow Start row of ROI
 * @param roiStartCol Start column of ROI
 * @param roiNumRows Number of rows in ROI
 * @param roiNumCols Number of columns in ROI
 * @param roiRowStride Stride of rows in ROI, determines row downsampling
 * @param roiColStride Stride of columns in ROI, determines column downsampling
 * @param output Output array
 * @param outputNumRows Number of rows of 'output'
 * @param outputNumCols Number of columns of 'output'
 * @param invalidVal Value to use for invalid values
 */
template <typename T, typename Validator>
__global__ void conditioned_downsampling_kernel(
    const T *input, const T *conditional, const unsigned int inputNumRows,
    const unsigned int inputNumCols, const unsigned int roiStartRow,
    const unsigned int roiStartCol, const unsigned int roiNumRows,
    const unsigned int roiNumCols, const unsigned int roiRowStride,
    const unsigned int roiColStride, T *output,
    const unsigned int outputNumRows, const unsigned int outputNumCols,
    const Validator validator) {
  // Assume 2D grid that covers ROI
  const unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ix >= roiNumCols || iy >= roiNumRows) {
    return;
  }

  // Compute index into input
  const unsigned int iRow = iy + roiStartRow;
  const unsigned int iCol = ix + roiStartCol;
  if (iRow >= inputNumRows || iCol >= inputNumCols) {
    return;
  }
  const size_t iIndex = iRow * inputNumCols + iCol;

  // Check if current thread is a valid downsample index
  if (ix % roiColStride == 0 && iy % roiRowStride == 0) {
    const unsigned int oRow = iy / roiRowStride;
    const unsigned int oCol = ix / roiColStride;
    if (oRow >= outputNumRows || oCol >= outputNumCols) {
      return;
    }
    const size_t oIndex = oRow * outputNumCols + oCol;
    // Check the condition
    if (validator.checkIsValid(input[iIndex], conditional[iIndex])) {
      output[oIndex] = input[iIndex];
    } else // Mark the value as invalid
      output[oIndex] = validator.getInvalidValue();
  }
}

// Here's a sample validator that checks only the condition value for a maximum
// value
template <typename T> struct MaximumValidator {
  T maxVal;

  __host__ __device__ bool checkIsValid(const T, const T cond) const {
    return cond < maxVal;
  }
  __host__ __device__ T getInvalidValue() const {
    return cuda::std::numeric_limits<T>::max();
  }
};

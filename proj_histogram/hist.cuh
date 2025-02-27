#include "sharedmem.cuh"

/**
 * @brief Gets the bin index for an input value.
 *        You are expected to check for out of range indices yourself.
 *        NOTE: due to floating-point inaccuracies, this may be off by 1 bin
 *        when using single-precision and/or evaluating values at bin edges.
 *        If exact values are required, enable use of validateBin().
 */
template <typename T>
__device__ int getBin(const T val, const T binSizeReciprocal,
                      const T firstBin) {
  // The reciprocal is used here since this can be pre-computed,
  // and may offer some speedup via a mul instruction instead of a div
  // instruction, especially in a loop like in the histogram kernel.
  return (val - firstBin) * binSizeReciprocal;
}

/**
 * @brief Validates the bin index against the actual bin boundaries.
 *        Due to floating point inaccuracies, the calculated bin in getBin()
 *        may be off by 1 bin. This will adjust the bin index accordingly.
 */
template <typename T>
__device__ int validateBin(const T val, const T binSize, const int binIdx) {
  const T binLeftLim = binIdx * binSize;
  // NOTE: do not change this to be binLeftLim + binSize, as that can be a
  // different floating point value!
  const T binRightLim = (binIdx + 1) * binSize;

  // Floating point precision resulted in overestimate
  if (val < binLeftLim)
    return binIdx - 1;

  // Floating point precision resulted in underestimate
  if (val >= binRightLim)
    return binIdx + 1;

  return binIdx;
}

/**
 * @brief Simple histogram kernel for equal length bins.
 *
 * @tparam T Type of inputs and bin edges
 * @param d_input Input array
 * @param inputLength Length of input array
 * @param d_hist int array of histogram counts
 * @param numBins Length of d_hist
 * @param binSize Size of a single bin
 * @param firstBin Value of the left-most bin
 * @param d_binIndices Optional output of the bin indices, for debugging
 */
template <typename T, bool correctBins = false>
__global__ void histogramKernel(const T *d_input, int inputLength, int *d_hist,
                                const int numBins, const T binSize,
                                const T firstBin, int *d_binIndices = nullptr) {
  // Pre-compute reciprocal of bin size
  const T binSizeReciprocal = 1.0 / binSize;

  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < inputLength;
       t += gridDim.x * blockDim.x) {
    // Retrieve input and get the initial bin index estimate
    auto input = d_input[t];
    int bin = getBin(input, binSizeReciprocal, firstBin);

    if constexpr (correctBins)
      bin = validateBin(input, binSize, bin);

    if (bin >= 0 && bin < numBins) {
      atomicAdd(&d_hist[bin], 1);
    }
    // Write the bin indices if they are requested (expected to be slower of
    // course, but useful for debugging)
    if (d_binIndices != nullptr) {
      d_binIndices[t] = bin;
    }
  }
}

#include "sharedmem.cuh"

/**
 * @brief Gets the bin index for an input value.
 *        You are expected to check for out of range indices yourself.
 */
template <typename T>
__device__ int getBin(const T val, const T binSize, const T firstBin) {
  return (val - firstBin) / binSize;
}

template <typename T>
__global__ void histogramKernel(const T *d_input, int inputLength, int *d_hist,
                                const int numBins, const T binSize,
                                const T firstBin) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < inputLength;
       t += gridDim.x * blockDim.x) {
    int bin = getBin(d_input[t], binSize, firstBin);
    if (bin >= 0 && bin < numBins) {
      atomicAdd(&d_hist[bin], 1);
    }
  }
}

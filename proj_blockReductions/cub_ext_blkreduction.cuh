#include "sharedmem.cuh"

/**
 * @brief Max and argmax reduction within a block. This does not currently
 * (24/4/2025) seem to have a simple cub implementation.
 *
 * @tparam T Type of input
 * @param sdata Input data in shared memory, assumed to have length blockDim.x
 * @param idx Associated indices of the input data
 */
template <typename T>
__device__ void blockReduceMaxAndArgMax(T *sdata, int *idx) {
  // We won't assume the user will sync before this.
  __syncthreads();

  // Reduce inside shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x] < sdata[threadIdx.x + s]) {
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
        idx[threadIdx.x] = idx[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
}

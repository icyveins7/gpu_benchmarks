#include "sharedmem.cuh"

static_assert(true); // dummy assert just for clangd LSP to stop complaining

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

/**
 * @brief Max and argmax reduction within a block. This does not currently
 * (24/4/2025) seem to have a simple cub implementation. The max and argmax are
 * in the index 0 element of each array at the end, as usual.
 *
 * @tparam T Type of input
 * @param sdata Input data in shared memory, assumed to have length blockDim.x
 * @param idx Associated indices of the input data in shared memory, same length
 */
template <typename T>
__device__ void blockReduceMaxAndArgMax(T *sdata, int *sidx) {
  // We won't assume the user will sync before this.
  __syncthreads();

  // Reduce inside shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sdata[threadIdx.x + s] >= sdata[threadIdx.x]) {
        // Prefer the later index if equal
        sidx[threadIdx.x] = sdata[threadIdx.x] == sdata[threadIdx.x + s]
                                ? max(sidx[threadIdx.x], sidx[threadIdx.x + s])
                                : sidx[threadIdx.x + s];
        sdata[threadIdx.x] = sdata[threadIdx.x + s];
      }
    }
    __syncthreads();
    // if (blockIdx.x == 25 && threadIdx.x < s)
    //   printf("Batch %d thread %d, [%d]: %d\n", s, threadIdx.x,
    //          sidx[threadIdx.x], sdata[threadIdx.x]);
  }
}

template <typename T>
__global__ void simpleBlockMaxAndArgMaxKernel(
    const T *in, const int inLenPerBlock, T *maxPerBlock, int *argMaxPerBlock,
    T *debugMaxPerBlock = nullptr, int *debugArgMaxPerBlock = nullptr) {
  // Get the input for the block
  const T *blk_in = &in[blockIdx.x * inLenPerBlock];

  // Get some shared mem for 1 block
  SharedMemory<T> smem;
  T *s_in = smem.getPointer();
  int *s_idx = (int *)&s_in[blockDim.x];
  // Initialize some starting values for each thread in the block
  s_in[threadIdx.x] = 0;
  s_idx[threadIdx.x] = -1;

  // Compare while loading
  for (int t = threadIdx.x; t < inLenPerBlock; t += blockDim.x) {
    T candidate = blk_in[t];
    if (candidate >= s_in[threadIdx.x]) {
      // Prefer the later index if equal
      s_idx[threadIdx.x] =
          candidate == s_in[threadIdx.x] ? max(s_idx[threadIdx.x], t) : t;
      s_in[threadIdx.x] = blk_in[t];
    }
  }

  // Debug shared memory states
  if (debugMaxPerBlock != nullptr) {
    debugMaxPerBlock[blockIdx.x * blockDim.x + threadIdx.x] = s_in[threadIdx.x];
  }
  if (debugArgMaxPerBlock != nullptr) {
    debugArgMaxPerBlock[blockIdx.x * blockDim.x + threadIdx.x] =
        s_idx[threadIdx.x];
  }

  blockReduceMaxAndArgMax(s_in, s_idx);

  if (threadIdx.x == 0) {
    maxPerBlock[blockIdx.x] = s_in[0];
    argMaxPerBlock[blockIdx.x] = s_idx[0];
  }
}

#pragma GCC diagnostic pop

#pragma once

#include "atomic_extensions.cuh"
#include "sharedmem.cuh"
#include <cuda/std/limits>

template <typename T>
__global__ void sumAndDownsampleMatrix(const T *input, T *output,
                                       const size_t width, const size_t height,
                                       const size_t widthDsr,
                                       const size_t heightDsr) {

  const size_t outWidth = width / widthDsr;
  const size_t outHeight = height / heightDsr;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int srcX, srcY;
  if (x < outWidth && y < outHeight) {
    T sum = 0;
    for (int i = 0; i < heightDsr; i++) {
      for (int j = 0; j < widthDsr; j++) {
        srcY = y * heightDsr + i;
        srcX = x * widthDsr + j;
        if (srcX < width && srcY < height)
          sum += input[srcY * width + srcX];
      }
    }
    output[y * outWidth + x] = sum;
  }
}

// Same as sumAndDownsampleMatrix but with threshold tracking
template <typename T, bool useExplicitWarpAggregation = false>
__global__ void sumAndDownsampleMatrixWithThreshold(
    const T *input, T *output, const size_t width, const size_t height,
    const size_t widthDsr, const size_t heightDsr, const T threshold,
    unsigned int *counter) {

  const size_t outWidth = width / widthDsr;
  const size_t outHeight = height / heightDsr;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int srcX, srcY;
  if (x < outWidth && y < outHeight) {
    T sum = 0;
    for (int i = 0; i < heightDsr; i++) {
      for (int j = 0; j < widthDsr; j++) {
        srcY = y * heightDsr + i;
        srcX = x * widthDsr + j;
        if (srcX < width && srcY < height)
          sum += input[srcY * width + srcX];
      }
    }
    output[y * outWidth + x] = sum;

    // Threshold checks
    if constexpr (useExplicitWarpAggregation) {
      if (sum > threshold)
        atomicAggInc(counter);
    } else {
      // Let nvcc handle it since it should automatically implement better code
      if (sum > threshold)
        atomicAdd(counter, 1);
    }
  }
}

template <typename T, typename Tidx> struct ValIdxPair {
  T val;
  Tidx idx;
};

template <typename T, typename U, typename V>
__global__ void sumAndDownsampleMatrixWithArgmaxBlockwiseKernel(
    const T *inputs, T *outputs, const int width, const int height,
    const unsigned int batch, const int widthDsr, const int heightDsr,
    V *d_maxargmax) {

  // Check block index
  unsigned int batchIdx = blockIdx.z;
  if (batchIdx > batch)
    return;

  // Load section into shared memory
  const unsigned int outWidth = width / widthDsr;
  const unsigned int outHeight = height / heightDsr;

  // Define the pointers for the block
  T *output = &outputs[batchIdx * outWidth * outHeight];
  const T *input = &inputs[batchIdx * width * height];
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t oIdx = y * outWidth + x;

  // Pixel is either completely valid or invalid, no 'half-way integrations'
  if (x >= outWidth || y >= outHeight)
    return;

  // Read original (oversampled) data into shared mem
  SharedMemory<T> smem;
  T *s_blkdata =
      smem.getPointer(); // blockDim.x * blockDim.y * widthDsr * heightDsr
  T *s_blkds = &s_blkdata[blockDim.x * blockDim.y * widthDsr *
                          heightDsr]; // blockDim.x * blockDim.y
  // Reading is done 1 tile at a time
  // TODO: this is currently wrong
  for (int j = 0; j < heightDsr; j++) {
    int sy = blockDim.y * j + threadIdx.y;
    for (int i = 0; i < widthDsr; i++) {
      int sx = blockDim.x * i + threadIdx.x;
      s_blkdata[sx + sy * blockDim.y] = input[y * width + x];
      printf("Loaded sx %d sy %d  from x %d y %d-> %u\n", sx, sy, x, y,
             s_blkdata[sx + sy * blockDim.y]);
    }
  }
  __syncthreads();

  // Shared-memory integration
  int sOidx = threadIdx.x +
              threadIdx.y * blockDim.x; // flattened 1d index within the block
  s_blkds[sOidx] = 0;
  for (int j = 0; j < heightDsr; j++) {
    int sy = threadIdx.y * heightDsr + j; // y read index
    for (int i = 0; i < widthDsr; i++) {
      int sx = threadIdx.x * widthDsr + i; // x read index
      s_blkds[sOidx] += s_blkdata[sx + sy * blockDim.x * widthDsr];
    }
  }
  __syncthreads();

  // Global writes out
  output[oIdx] = s_blkds[sOidx];

  // Continue work to find max and argmax inside the shared mem
  U argmax_thread = sOidx;
  for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
    if (sOidx < s) {
      if (s_blkds[sOidx + s] >= s_blkds[sOidx]) {
        s_blkds[sOidx] = s_blkds[sOidx + s];
        argmax_thread = sOidx + s;
      }
    }
    __syncthreads();
  }
  // Thread 0 has max and argmax
  if (threadIdx.x == 0) {
    // Translate block's argmax into image-space index
    int sxmax = argmax_thread % blockDim.x;
    int symax = argmax_thread / blockDim.x;
    int xmax = sxmax + blockIdx.x * blockDim.x;
    int ymax = symax + blockIdx.y * blockDim.y;

    // Craft the squeezed output
    V squeezed = squeezeValueIndexForAtomic<T, U, V>(s_blkds[0],
                                                     (U)(xmax + ymax * width));
    // Atomically update
    atomicMax(&d_maxargmax[batchIdx], squeezed);
  }
}

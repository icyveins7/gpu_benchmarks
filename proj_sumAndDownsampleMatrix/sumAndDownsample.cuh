#pragma once

#include "accessors.cuh"
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

template <typename T>
__device__ void blockRoiIntegrate(const T *src, const int srcWidth,
                                  const int srcHeight, const int widthDsr,
                                  const int heightDsr, T *dst) {
  // Implicitly expects destination has sufficient and exact dimensions
  // for the downsampled output (srcWidth / widthDsr, srcHeight / heightDsr)
  int dstWidth = srcWidth / widthDsr;
  int dstHeight = srcHeight / heightDsr;
  // Each thread loops over destination points; this prevents race conditions on
  // writes First 2 loops are simply a block-stride over the destination
  for (int ty = threadIdx.y; ty < dstHeight; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < dstWidth; tx += blockDim.x) {
      // Next 2 loops are to access the source read locations to accumulate sum
      // from
      for (int i = 0; i < heightDsr; i++) {
        int srcY = ty * heightDsr + i;
        for (int j = 0; j < widthDsr; j++) {
          int srcX = tx * widthDsr + j;
          dst[ty * dstWidth + tx] += src[srcY * srcWidth + srcX];
        }
      }
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
  // these are output x, y positions
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t oIdx = y * outWidth + x;

  // NOTE: you should not disable or return here if the threads don't have a
  // 'valid' output index, since we need all threads to perform other stuff in
  // the block, and returning here would cause issues with block-wise
  // loads/reductions etc

  // Read original (oversampled) data into shared mem
  SharedMemory<T> smem;
  T *s_blkdata =
      smem.getPointer(); // blockDim.x * blockDim.y * widthDsr * heightDsr
  T *s_blkds = &s_blkdata[blockDim.x * blockDim.y * widthDsr *
                          heightDsr]; // blockDim.x * blockDim.y
  // Define the ROI for the block
  int roiStartX = blockIdx.x * blockDim.x * widthDsr;
  int roiStartY = blockIdx.y * blockDim.y * heightDsr;
  int roiLengthX = blockDim.x * widthDsr;
  int roiLengthY = blockDim.y * heightDsr;
  // Handle edge of image remnants, and ensure ROI is always a multiple of
  // downsample rate
  roiLengthX =
      min(roiLengthX, (width - roiStartX) - (width - roiStartX) % widthDsr);
  roiLengthY =
      min(roiLengthY, (height - roiStartY) - (height - roiStartY) % heightDsr);

  // Reading is done 1 tile at a time
  blockRoiLoad<T>(input, width, height, roiStartX, roiLengthX, roiStartY,
                  roiLengthY, s_blkdata);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int i = 0; i < roiLengthX * roiLengthY; i++)
      printf("[%d/%d]: %d\n", i, roiLengthX * roiLengthY, s_blkdata[i]);
  }
  __syncthreads();

  // Shared-memory integration
  // zero out first
  int dsLengthX = roiLengthX / widthDsr;
  int dsLengthY = roiLengthY / heightDsr;
  for (int ty = threadIdx.y; ty < dsLengthY; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < dsLengthX; tx += blockDim.x) {
      s_blkds[ty * dsLengthX + tx] = 0;
    }
  }
  blockRoiIntegrate<T>(s_blkdata, roiLengthX, roiLengthY, widthDsr, heightDsr,
                       s_blkds);

  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int i = 0; i < dsLengthX * dsLengthY; i++)
      printf("[%d/%d]: %d\n", i, dsLengthX * dsLengthY, s_blkds[i]);
  }

  // Global writes out
  int sOidx = threadIdx.x + threadIdx.y * roiLengthX / widthDsr;
  if (x < outWidth && y < outHeight)
    output[oIdx] = s_blkds[sOidx];
  __syncthreads(); // make sure everyone wrote out before we do our reduction

  // Continue work to find max and argmax inside the shared mem
  // We reuse shared memory from the original block data for the indices,
  // this is usually sufficient
  // TODO: add static_assert for size ?
  int *s_idx = (int *)s_blkdata;
  // Label each index value in shared memory
  for (int t = threadIdx.x + threadIdx.y * dsLengthX; t < dsLengthX * dsLengthY;
       t += blockDim.x * blockDim.y)
    s_idx[t] = t;

  for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
    if (sOidx < s && sOidx < dsLengthX * dsLengthY &&
        sOidx + s < dsLengthX * dsLengthY) {
      if (s_blkds[sOidx + s] >= s_blkds[sOidx]) {
        printf("sOidx: %d s: %d, values: %d %d\n", sOidx, s, s_blkds[sOidx],
               s_blkds[sOidx + s]);
        s_blkds[sOidx] = s_blkds[sOidx + s];
        s_idx[sOidx] = s_idx[sOidx + s];
      }
    }
    __syncthreads();
  }
  // Thread 0 has max and argmax
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    // Translate block's argmax into image-space index
    int sxmax = s_idx[0] % dsLengthX;
    int symax = s_idx[0] / dsLengthX;
    int xmax = sxmax + blockIdx.x * blockDim.x;
    int ymax = symax + blockIdx.y * blockDim.y;

    // Craft the squeezed output
    printf("Blk %d,%d Max: %d at %d, %d [thread %d] -> %d, %d\n", blockIdx.x,
           blockIdx.y, s_blkds[0], sxmax, symax, s_idx[0], xmax, ymax);
    V squeezed = squeezeValueIndexForAtomic<T, U, V>(s_blkds[0],
                                                     (U)(xmax + ymax * width));
    // Atomically update
    atomicMax(&d_maxargmax[batchIdx], squeezed);
  }
}

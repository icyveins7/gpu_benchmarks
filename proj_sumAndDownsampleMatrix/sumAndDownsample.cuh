#include "atomic_extensions.cuh"

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

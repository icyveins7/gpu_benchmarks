#pragma once

#include <cstddef>
#include <cstdint>
#include <vector_types.h>

#include "containers/image.cuh"
#include "sharedmem.cuh"

/**
 * @brief Applies a naive zero-padded 2D median filter.
 *
 * @details Each thread processes output pixels using a 2D grid-stride loop and
 * stores one complete window in its own row of dynamic shared memory. The
 * median is found with an intentionally slow insertion sort. For a window with
 * N = windowDims.x * windowDims.y elements, sorting each pixel is O(N^2).
 * Both window dimensions must be nonzero and odd; otherwise the kernel returns
 * without writing output. The launch must provide at least
 * blockDim.x * blockDim.y * N * sizeof(T) bytes of dynamic shared memory.
 *
 * @tparam T Input and output element type.
 * @param input Input image.
 * @param output Output image.
 * @param windowDims Median window width in x and height in y.
 */
template <typename T = int32_t>
__global__ void median2d_naive_kernel(const containers::Image<const T> input,
                                      containers::Image<T> output,
                                      const uint2 windowDims) {
  if (windowDims.x == 0 || windowDims.y == 0 || windowDims.x % 2 == 0 ||
      windowDims.y % 2 == 0)
    return;

  const size_t windowArea = static_cast<size_t>(windowDims.x) * windowDims.y;
  const unsigned int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  SharedMemory<T> smem;
  T *window = smem.getPointer() + threadIndex * windowArea;

  for (int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
       outputRow < output.height; outputRow += blockDim.y * gridDim.y) {
    for (int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
         outputCol < output.width; outputCol += blockDim.x * gridDim.x) {
      const int firstInputRow =
          outputRow - static_cast<int>(windowDims.y / 2);
      const int firstInputCol =
          outputCol - static_cast<int>(windowDims.x / 2);

      size_t windowIndex = 0;
      for (unsigned int windowRow = 0; windowRow < windowDims.y; ++windowRow) {
        const int inputRow = firstInputRow + static_cast<int>(windowRow);
        for (unsigned int windowCol = 0; windowCol < windowDims.x;
             ++windowCol) {
          const int inputCol = firstInputCol + static_cast<int>(windowCol);
          window[windowIndex++] =
              input.rowIsValid(inputRow) && input.colIsValid(inputCol)
                  ? input.at(inputRow, inputCol)
                  : T{0};
        }
      }

      for (size_t i = 1; i < windowArea; ++i) {
        const T value = window[i];
        size_t j = i;
        while (j > 0 && window[j - 1] > value) {
          window[j] = window[j - 1];
          --j;
        }
        window[j] = value;
      }

      output.at(outputRow, outputCol) = window[windowArea / 2];
    }
  }
}

#pragma once
/*
Cropping operations are often just a small step, taken after some other point
selection routine. As such it is usually more useful to have them as a device
function rather than only as a kernel, as this will make it easier for them to
be included as part of another kernel.
*/

/**
 * @brief This is a simple grid-stride implementation that crops around a target
 * point. It specifically enforces the cropped dimensions for the cases which
 * clip the borders, by moving the edges to the edge of the image (priority for
 * dimensions over crop directions).
 *
 * The crop directions are specified using an int4 which is defined as follows:
 *     .x : pixels to left of target (must be positive value)
 *     .y : pixels to right of target
 *     .z : pixels to top of target (must be positive value)
 *     .w : pixels to bottom of target
 *
 * Note that this uses the convention where y increases downwards.
 *
 * @example
 *     X O O O O O X X
 *     X O O O O O X X --> ptCoords (*) = (2, 2)
 *     X O * O O O X X --> cropDirections (O) = (1, 3, 2, 1)
 *     X O O O O O X X
 *     X X X X X X X X
 *     X X X X X X X X
 *     X X X X X X X X
 *     X X X X X X X X
 *
 * @tparam T Type of data
 * @param image Source
 * @param width Source width
 * @param height Source height
 * @param cropDirections int4 containing the 4 respective distances from target
 * pixel
 * @param ptCoords int2 containing the target pixel.
 * @param dst Destination
 */
template <typename T>
__device__ void cropAroundPoint_gridStride(const T *image, const int width,
                                           const int height,
                                           const int4 cropDirections,
                                           const int2 ptCoords, T *dst) {

  // First we check that it's not going to clip the border
  int left = ptCoords.x - cropDirections.x;
  int right = ptCoords.x + cropDirections.y; // inclusive
  if (left < 0) {
    left = 0;
    right = cropDirections.x + cropDirections.y;
  } else if (right >= width) {
    right = width - 1;
    left = width - 1 - cropDirections.x - cropDirections.y;
  }

  // We use convention that top is lower index than bottm i.e. y goes downwards
  int top = ptCoords.y - cropDirections.z;
  int bottom = ptCoords.y + cropDirections.w; // inclusive
  if (top < 0) {
    top = 0;
    bottom = cropDirections.z + cropDirections.w;
  } else if (bottom >= height) {
    bottom = height - 1;
    top = height - 1 - cropDirections.z - cropDirections.w;
  }

  // Compute the expected destination dimensions
  int dstWidth = right - left + 1;
  int dstHeight = bottom - top + 1;

  // Simple grid-stride over destination to copy the cropped image
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < dstWidth * dstHeight;
       t += blockDim.x * gridDim.x) {
    int dstX = t % dstWidth;
    int dstY = t / dstWidth;

    int srcX = left + dstX;
    int srcY = top + dstY;
    int srcIdx = srcY * width + srcX;

    dst[t] = image[srcIdx];
  }
}

/**
 * @brief Look at cropAroundPoint_gridStride device kernel for details.
 */
template <typename T>
__global__ void
cropAroundPoint_gridStrideKernel(const T *image, const int width,
                                 const int height, const int4 cropDims,
                                 const int2 ptCoords, T *dst) {
  // Redirect to the device function entirely
  cropAroundPoint_gridStride(image, width, height, cropDims, ptCoords, dst);
}

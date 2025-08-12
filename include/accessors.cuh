#pragma once

#include <stdio.h>

/**
 * @brief Converts ROI coordinates to the original source image coordinates.
 *
 * @param roiX ROI X coordinate
 * @param roiY ROI Y coordinate
 * @param roiStartX ROI start X coordinate in the source image
 * @param roiStartY ROI start Y coordinate in the source image
 * @param x Output X coordinate in the source image
 * @param y Output Y coordinate in the source image
 */
__device__ void fromRoiCoords(const int roiX, const int roiY,
                              const int roiStartX, const int roiStartY, int &x,
                              int &y) {
  x = roiX + roiStartX;
  y = roiY + roiStartY;
}

template <typename T>
__device__ void threadRoiLoad(const T *src, const int srcWidth,
                              const int srcHeight, const int roiStartX,
                              const int roiStartY, T &val, int tx = -1,
                              int ty = -1) {
  tx = (tx == -1) ? threadIdx.x : tx;
  ty = (ty == -1) ? threadIdx.y : ty;
  int x, y;
  fromRoiCoords(tx, ty, roiStartX, roiStartY, x, y);
  if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight)
    val = src[y * srcWidth + x];
}

/**
 * @brief Loads an ROI from a source image to a packed destination, whose
 * dimensions equal the ROI.
 *
 * @details This function is agnostic of the block dimensions, and will
 * grid-stride over the ROI dimensions to fully load it into the destination. If
 * the goal is to have only one load per block, the programmer should ensure
 * that the grid adequately covers the ROI. This loader is both source and
 * destination dimensions-aware, meaning it will not read from the source or
 * write to the destination if either source or destination indices exceed their
 * respective heights/widths.
 *
 * @example
 * Below is an example of the ROI load, where the image is presented top-down
 * i.e. row 0 is the top-most row.
 *    srcWidth = 9
 *    srcHeight = 4
 *    roiStartX = 3
 *    roiStartY = 1
 *    roiLengthX = 3
 *    roiLengthY = 2
 *
 * XXXXXXXXX
 * XXXABCXXX    ->   ABC
 * XXXDEFXXX    ->   DEF
 * XXXXXXXXX
 *
 * @tparam T Type of data
 * @param src Source image
 * @param srcWidth Source image width
 * @param srcHeight Source image height
 * @param roiStartX Starting X pixel of ROI
 * @param roiLengthX Length of ROI in X
 * @param roiStartY Starting Y pixel of ROI
 * @param roiLengthY Length of ROI in Y
 * @param dst Destination, can be global or shared
 */
template <typename T>
__device__ void gridRoiLoad(const T *src, const int srcWidth,
                            const int srcHeight, const int roiStartX,
                            const int roiLengthX, const int roiStartY,
                            const int roiLengthY, T *dst) {
  // Grid-stride over the destination; tx and ty are the roi-zeroed coords
  int x, y;
  for (int ty = threadIdx.y + blockDim.y * blockIdx.y; ty < roiLengthY;
       ty += blockDim.y * gridDim.y) {
    for (int tx = threadIdx.x + blockDim.x * blockIdx.x; tx < roiLengthX;
         tx += blockDim.x * gridDim.x) {
      fromRoiCoords(tx, ty, roiStartX, roiStartY, x, y);
      if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight)
        dst[ty * roiLengthX + tx] = src[y * srcWidth + x];
    }
  }
}

/**
 * @brief Loads an ROI from a source image to a packed destination, whose
 * dimensions equal the ROI. Usually used for shared memory, or any other
 * operation involving only one block.
 *
 * @detail This block-specific loader is useful in contexts where each CUDA
 * block works on its own ROI/tile. This loader is both source and destination
 * dimensions-aware, meaning it will not read from the source or write to the
 * destination if either source or destination indices exceed their respective
 * heights/widths.
 *
 * NOTE: this expects that all threads in the block are active at the point of
 * invocation. The programmer must remember not to do an early 'return' at the
 * start of the kernel for some threads, even if they are not used for later
 * results, as this will affect the block-wise iteration over the ROI i.e. some
 * values will be missed.
 *
 * @example
 * Below is an example of the ROI load, where the image is presented top-down
 * i.e. row 0 is the top-most row.
 *    srcWidth = 9
 *    srcHeight = 4
 *    roiStartX = 3
 *    roiStartY = 1
 *    roiLengthX = 3
 *    roiLengthY = 2
 *
 * XXXXXXXXX
 * XXXABCXXX    ->   ABC
 * XXXDEFXXX    ->   DEF
 * XXXXXXXXX
 *
 * @tparam T Type of data
 * @param src Source image
 * @param srcWidth Source image width
 * @param srcHeight Source image height
 * @param roiStartX Starting X pixel of ROI
 * @param roiLengthX Length of ROI in X
 * @param roiStartY Starting Y pixel of ROI
 * @param roiLengthY Length of ROI in Y
 * @param dst Destination, can be global or shared
 */
template <typename T>
__device__ void blockRoiLoad(const T *src, const int srcWidth,
                             const int srcHeight, const int roiStartX,
                             const int roiLengthX, const int roiStartY,
                             const int roiLengthY, T *dst) {
  // Block-stride over the destination
  for (int ty = threadIdx.y; ty < roiLengthY; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < roiLengthX; tx += blockDim.x) {
      threadRoiLoad(src, srcWidth, srcHeight, roiStartX, roiStartY,
                    dst[ty * roiLengthX + tx], tx, ty);
    }
  }
}

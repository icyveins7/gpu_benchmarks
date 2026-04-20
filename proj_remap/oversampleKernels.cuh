#pragma once

#include <cmath>
#include <type_traits>

#include <cuda/std/cmath>

#include "block_and_grid_sizing.cuh"
#include "containers/image.cuh"
#include "extra_type_traits.cuh"
#include "sharedmem.cuh"

#include "kernels.h"

template <typename Tin, typename Tout, typename Tcalc = float,
          bool useSharedMemForInput = false>
__global__ void oversampleBilerpAndCombineKernel(
    containers::Image<const Tin> in, containers::Image<Tout> out,
    const int2 oversampleFactor, // assumed odd
    const cuda_vec2_t<Tcalc> outOffset, const cuda_vec2_t<Tcalc> outStep) {
  static_assert(std::is_floating_point<Tcalc>::value,
                "Tcalc must be a floating point type");
  // Calculate number of blocks required for output
  int2 numBlocks = make_int2(justEnoughBlocks(blockDim.x, out.width),
                             justEnoughBlocks(blockDim.y, out.height));
  // Calculate number of padded elements
  // Assumes oversampling is already odd
  int2 padding = make_int2(oversampleFactor.x / 2, oversampleFactor.y / 2);
  // Calculate the 'length' in input coordinate space per oversampled element
  /*
  Example, for oversampleFactor 3
  O X O|O X O|O X O
    |     |     |
    |     outStep
    |
    centre of output pixels
  */
  Tcalc overSampStepX = outStep.x / oversampleFactor.x;
  Tcalc overSampStepY = outStep.y / oversampleFactor.y;
  // And then use this to calculate the first oversampled coordinate (in the
  // padding)
  Tcalc oversampStartX = outOffset.x - padding.x * overSampStepX;
  Tcalc oversampStartY = outOffset.y - padding.y * overSampStepY;
  // TODO: maybe we can move some of these outside

  // Allocate shared memory
  SharedMemory<Tcalc> smem;
  containers::Image<Tcalc> s_oversamp{smem.getPointer(),
                                      blockDim.x * oversampleFactor.x,
                                      blockDim.y * oversampleFactor.y};

  // // Optionally also use shared memory for input tile
  // containers::Image<Tin> s_intile{nullptr, 0, 0};
  // if constexpr (useSharedMemForInput) {
  //   s_intile.data = &s_oversamp.data[s_oversamp.width * s_oversamp.height];
  //   // TODO: fill width/height based on expected tile start/stop
  // }

  // Loop over blocks
  for (int i = blockIdx.y; i < numBlocks.y; i += gridDim.y) {
    for (int j = blockIdx.x; j < numBlocks.x; j += gridDim.x) {
      // // Fill shared memory for input tile (if used)
      // if constexpr (useSharedMemForInput) {
      //   for (int ty = threadIdx.y; ty < s_intile.height; ty += blockDim.y) {
      //     for (int tx = threadIdx.x; tx < s_intile.width; tx += blockDim.x) {
      //       // TODO:
      //     }
      //   }
      //
      //   __syncthreads();
      // }

      // Begin calculation of oversampled tile
      int oversampBlkStartYIdx = i * s_oversamp.height;
      int oversampBlkStartXIdx = j * s_oversamp.width;
      // if (i > 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf("i = %d, j = %d, oversampBlkStartIdx %d, %d\n", i, j,
      //          oversampBlkStartYIdx, oversampBlkStartXIdx);
      // }
      for (int y = threadIdx.y; y < s_oversamp.height; y += blockDim.y) {
        Tcalc sy = (oversampBlkStartYIdx + y) * overSampStepY + oversampStartY;
        for (int x = threadIdx.x; x < s_oversamp.width; x += blockDim.x) {
          Tcalc sx =
              (oversampBlkStartXIdx + x) * overSampStepX + oversampStartX;
          // sx, sy are the coordinates to bilerp

          // we get the nearest integer index to read from
          int iy = (int)cuda::std::floor(sy);
          int ix = (int)cuda::std::floor(sx);

          // if (i > 0 && j == 0 && y < 3 && x < 3) {
          //   printf("blk %d, %d -> read from %d, %d\n", i, j, iy, ix);
          // }

          // Extract the 4 corners of data
          // Here is where you would include logic to handle pixel reading
          // For now we just read simply and default to 0 if it doesn't exist
          // Read and cast to out floating point type
          Tcalc topLeft = in.atWithDefault(iy, ix);
          Tcalc topRight = in.atWithDefault(iy, ix + 1);
          Tcalc botLeft = in.atWithDefault(iy + 1, ix);
          Tcalc botRight = in.atWithDefault(iy + 1, ix + 1);

          Tcalc interpolated = bilinearInterpolate<Tcalc>(
              topLeft, topRight, botLeft, botRight, sx, sy);
          // if (i > 0 && j == 0 && y < 3 && x < 3) {
          //   printf(
          //       "sy %f, sx %f -> 4 values are %f, %f, %f, %f -> interp to
          //       %f\n", sy, sx, topLeft, topRight, botLeft, botRight,
          //       interpolated);
          // }

          // Write to shared memory
          s_oversamp.at(y, x) = interpolated;
        }
      }
      __syncthreads(); // complete fill of shared mem oversampled tile

      // Sum over the oversampled tile for each thread's requirements
      Tcalc value = 0;
      for (int oy = 0; oy < oversampleFactor.y; ++oy) {
        for (int ox = 0; ox < oversampleFactor.x; ++ox) {
          value += s_oversamp.at(threadIdx.y * oversampleFactor.y + oy,
                                 threadIdx.x * oversampleFactor.x + ox);
          // if (i == 0 && j == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
          //   printf("value + %f(from %d, %d) = %f\n",
          //          s_oversamp.at(threadIdx.y * oversampleFactor.y + oy,
          //                        threadIdx.x * oversampleFactor.x + ox),
          //          threadIdx.y * oversampleFactor.y + oy,
          //          threadIdx.x * oversampleFactor.x + ox, value);
          // }
        }
      }
      // Write out
      int outRow = i * blockDim.y + threadIdx.y;
      int outCol = j * blockDim.x + threadIdx.x;
      if (out.rowIsValid(outRow) && out.colIsValid(outCol))
        out.at(outRow, outCol) = (Tout)(value);
    }
  }
}

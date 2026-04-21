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
          bool useSharedMem = true>
__global__ void oversampleBilerpAndCombineKernel(
    containers::Image<const Tin> in, containers::Image<Tout> out,
    const int2 oversampleFactor, // assumed odd
    const cuda_vec2_t<Tcalc> outOffset, const cuda_vec2_t<Tcalc> outStep,
    // NOTE: Surprisingly, ncu says computing numOutBlocks stalls a lot, and in
    // fact just passing this in directly speeds up about 4%
    const int2 numOutBlocks) {
  static_assert(std::is_floating_point<Tcalc>::value,
                "Tcalc must be a floating point type");

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

  // Allocate shared memory if needed
  SharedMemory<Tcalc> smem;
  containers::Image<Tcalc> s_oversamp(nullptr, 0, 0);
  if constexpr (useSharedMem) {
    s_oversamp.initialize(smem.getPointer(), blockDim.x * oversampleFactor.x,
                          blockDim.y * oversampleFactor.y);
  }

  // Loop over blocks
  for (int i = blockIdx.y; i < numOutBlocks.y; i += gridDim.y) {
    for (int j = blockIdx.x; j < numOutBlocks.x; j += gridDim.x) {

      // Begin calculation of oversampled tile
      int oversampBlkStartYIdx = i * oversampleFactor.y * blockDim.y;
      int oversampBlkStartXIdx = j * oversampleFactor.x * blockDim.x;
      // if (i > 0 && threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf("i = %d, j = %d, oversampBlkStartIdx %d, %d\n", i, j,
      //          oversampBlkStartYIdx, oversampBlkStartXIdx);
      // }

      // Define final output to aggregate over
      Tcalc value = 0;

      // If using shared mem to calculate the oversampled
      if constexpr (useSharedMem) {
        for (int y = threadIdx.y; y < s_oversamp.height; y += blockDim.y) {
          Tcalc sy =
              (oversampBlkStartYIdx + y) * overSampStepY + oversampStartY;
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
      } // end of method 1, using shared mem for oversampled workspace
      else {
        for (int oy = 0; oy < oversampleFactor.y; ++oy) {
          // Define y index for this block/thread
          int y = threadIdx.y * oversampleFactor.y + oy;
          for (int ox = 0; ox < oversampleFactor.x; ++ox) {
            int x = threadIdx.x * oversampleFactor.x + ox;
            // Compute interpolated value directly
            Tcalc sy =
                (oversampBlkStartYIdx + y) * overSampStepY + oversampStartY;
            Tcalc sx =
                (oversampBlkStartXIdx + x) * overSampStepX + oversampStartX;
            // sx, sy are the coordinates to bilerp

            // we get the nearest integer index to read from
            int iy = (int)cuda::std::floor(sy);
            int ix = (int)cuda::std::floor(sx);

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

            value += interpolated;
          }
        }
      } // end of method 2, direct calculation and register accumulation

      // Write out final output
      int outRow = i * blockDim.y + threadIdx.y;
      int outCol = j * blockDim.x + threadIdx.x;
      if (out.rowIsValid(outRow) && out.colIsValid(outCol))
        out.at(outRow, outCol) = (Tout)(value);
    }
  }
}

template <typename Tin, typename Tout, typename Tcalc = float,
          bool useSharedMem = true>
void oversampleBilerpAndCombine(containers::Image<const Tin> in,
                                containers::Image<Tout> out,
                                const int2 oversampleFactor, // assumed odd
                                const cuda_vec2_t<Tcalc> outOffset,
                                const cuda_vec2_t<Tcalc> outStep,
                                const dim3 tpb, cudaStream_t stream = 0) {
  // Throw if not odd oversample
  if (oversampleFactor.x % 2 == 0 || oversampleFactor.y % 2 == 0) {
    throw std::runtime_error("oversampleFactor must be odd in both dimensions");
  }

  // Compute number of blocks to cover the output
  dim3 outblks(justEnoughBlocks(tpb.x, out.width),
               justEnoughBlocks(tpb.y, out.height));
  size_t shmem = 0;
  if constexpr (useSharedMem) {
    shmem =
        tpb.x * tpb.y * oversampleFactor.x * oversampleFactor.y * sizeof(Tcalc);
  }

  // Possibly adjust the number of blocks of grid to not cover output
  dim3 blks(outblks.x,
            outblks.y); // early testing shows possibly 1-2% speedup if we use
                        // 1/2 total blocks for example, probably only relevant
                        // when number of blocks is large

  // printf("Using shared mem %zu bytes\n", shmem);
  oversampleBilerpAndCombineKernel<int, float, Tcalc, useSharedMem>
      <<<blks, tpb, shmem, stream>>>(in, out, oversampleFactor, outOffset,
                                     outStep,
                                     int2{(int)outblks.x, (int)outblks.y});
}

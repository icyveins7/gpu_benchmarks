#pragma once

#include <cmath>
#include <type_traits>

#include <cuda/std/cmath>

#include "block_and_grid_sizing.cuh"
#include "containers/image.cuh"
#include "extra_type_traits.cuh"
#include "sharedmem.cuh"

#include "kernels.h"

/**
 * @brief Primarily used to pre-calculate constants for the oversampling, so the
 * kernel doesn't need to do it at the start.
 */
template <typename Tcalc> struct OversampleKernelParams {
  Tcalc overSampStepX;
  Tcalc overSampStepY;
  Tcalc overSampStartX;
  Tcalc overSampStartY;

  OversampleKernelParams(const int2 oversampleFactor,
                         const cuda_vec2_t<Tcalc> outStep,
                         const cuda_vec2_t<Tcalc> outOffset) {
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
    overSampStepX = outStep.x / oversampleFactor.x;
    overSampStepY = outStep.y / oversampleFactor.y;
    // And then use this to calculate the first oversampled coordinate (in the
    // padding)
    overSampStartX = outOffset.x - padding.x * overSampStepX;
    overSampStartY = outOffset.y - padding.y * overSampStepY;
  }
};

template <typename Tin, typename Tout, typename Tcalc = float,
          bool useSharedMem = true>
__global__ void oversampleBilerpAndCombineKernel(
    containers::Image<const Tin> in, containers::Image<Tout> out,
    const int2 oversampleFactor, // assumed odd
    const OversampleKernelParams<Tcalc> params,
    // NOTE: Surprisingly, ncu says computing numOutBlocks stalls a lot, and in
    // fact just passing this in directly speeds up about 4%
    const int2 numOutBlocks) {
  static_assert(std::is_floating_point<Tcalc>::value,
                "Tcalc must be a floating point type");

  SharedMemory<Tin> smem;
  containers::ImageTile<Tin> stile(nullptr, 0, 0, 0, 0);
  // We don't initialize it here, since the starting tile row/col derivation
  // happens in the blockwise loop below

  // Loop over blocks
  for (int i = blockIdx.y; i < numOutBlocks.y; i += gridDim.y) {
    for (int j = blockIdx.x; j < numOutBlocks.x; j += gridDim.x) {
      // Begin calculation of oversampled tile
      int oversampBlkStartYIdx = i * oversampleFactor.y * blockDim.y;
      int oversampBlkStartXIdx = j * oversampleFactor.x * blockDim.x;

      // Define final output to aggregate over
      Tcalc value = 0;

      if constexpr (useSharedMem) {
        // Compute the starting tile position and dimensions
        Tcalc tileStartY =
            oversampBlkStartYIdx * params.overSampStepY + params.overSampStartY;
        Tcalc tileStartX =
            oversampBlkStartXIdx * params.overSampStepX + params.overSampStartX;

        // int next_oversampBlkStartYIdx =
        //     (i + 1) * oversampleFactor.y * blockDim.y;
        // int next_oversampBlkStartXIdx =
        //     (j + 1) * oversampleFactor.x * blockDim.x;
        // Tcalc tileEndY =
        //     (next_oversampBlkStartYIdx - 1) * params.overSampStepY +
        //     params.overSampStartY;
        // Tcalc tileEndX =
        //     (next_oversampBlkStartXIdx - 1) * params.overSampStepX +
        //     params.overSampStartX;

        // NOTE: this is technically fixed
        int tilewidth =
            (oversampleFactor.x * blockDim.x) * params.overSampStepX +
            2; // +1 both sides for safety
        int tileheight =
            (oversampleFactor.y * blockDim.y) * params.overSampStepY +
            2; // +1 both sides for safety

        stile.initialize(
            smem.getPointer(), tilewidth, tileheight,
            cuda::std::floor(
                tileStartY), // these may be negative, so you must floor
            cuda::std::floor(tileStartX) // you cannot directly cast to int
        );
        stile.fillFromImage(in, 0);
        __syncthreads();

        // Now we have a tile in shared memory, we simply reference the tile
        // instead of the image when doing the interpolations
        for (int oy = 0; oy < oversampleFactor.y; ++oy) {
          // Define y index for this block/thread
          int y = threadIdx.y * oversampleFactor.y + oy;
          for (int ox = 0; ox < oversampleFactor.x; ++ox) {
            int x = threadIdx.x * oversampleFactor.x + ox;
            // Compute interpolated value directly
            Tcalc sy = (oversampBlkStartYIdx + y) * params.overSampStepY +
                       params.overSampStartY;
            Tcalc sx = (oversampBlkStartXIdx + x) * params.overSampStepX +
                       params.overSampStartX;
            // sx, sy are the coordinates to bilerp

            // we get the nearest integer index to read from
            int iy = (int)cuda::std::floor(sy);
            int ix = (int)cuda::std::floor(sx);

            // Extract the 4 corners of data
            // Here is where you would include logic to handle pixel reading
            // For now we just read simply and default to 0 if it doesn't exist
            // Read and cast to out floating point type
            Tcalc topLeft = stile.at(iy, ix);
            Tcalc topRight = stile.at(iy, ix + 1);
            Tcalc botLeft = stile.at(iy + 1, ix);
            Tcalc botRight = stile.at(iy + 1, ix + 1);

            Tcalc interpolated = bilinearInterpolate<Tcalc>(
                topLeft, topRight, botLeft, botRight, sx, sy);

            value += interpolated;
          }
        }
      } else {
        for (int oy = 0; oy < oversampleFactor.y; ++oy) {
          // Define y index for this block/thread
          int y = threadIdx.y * oversampleFactor.y + oy;
          for (int ox = 0; ox < oversampleFactor.x; ++ox) {
            int x = threadIdx.x * oversampleFactor.x + ox;
            // Compute interpolated value directly
            Tcalc sy = (oversampBlkStartYIdx + y) * params.overSampStepY +
                       params.overSampStartY;
            Tcalc sx = (oversampBlkStartXIdx + x) * params.overSampStepX +
                       params.overSampStartX;
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
      }

      // Write out final output
      int outRow = i * blockDim.y + threadIdx.y;
      int outCol = j * blockDim.x + threadIdx.x;
      if (out.rowIsValid(outRow) && out.colIsValid(outCol))
        out.at(outRow, outCol) =
            (Tout)(value / (oversampleFactor.x * oversampleFactor.y));
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

  // Compute oversample kernel parameters externally
  OversampleKernelParams<Tcalc> params(oversampleFactor, outStep, outOffset);

  // Compute number of blocks to cover the output
  dim3 outblks(justEnoughBlocks(tpb.x, (unsigned int)out.width),
               justEnoughBlocks(tpb.y, (unsigned int)out.height));
  size_t shmem = 0;

  // DEPRECATED. LATER CHANGE WHEN NEEDED FOR INPUT TILE SHMEM
  if constexpr (useSharedMem) {
    // NOTE: this is technically fixed
    int tilewidth = (oversampleFactor.x * tpb.x) * params.overSampStepX +
                    2; // +1 both sides for safety
    int tileheight = (oversampleFactor.y * tpb.y) * params.overSampStepY +
                     2; // +1 both sides for safety
    shmem = tilewidth * tileheight * sizeof(Tin);
    printf("shmem tile is %dx%d, %zu bytes\n", tilewidth, tileheight, shmem);
  }

  // Possibly adjust the number of blocks of grid to not cover output
  dim3 blks(outblks.x,
            outblks.y); // early testing shows possibly 1-2% speedup if we use
                        // 1/2 total blocks for example, probably only relevant
                        // when number of blocks is large

  // printf("Using shared mem %zu bytes\n", shmem);
  oversampleBilerpAndCombineKernel<int, float, Tcalc, useSharedMem>
      <<<blks, tpb, shmem, stream>>>(in, out, oversampleFactor, params,
                                     int2{(int)outblks.x, (int)outblks.y});
}

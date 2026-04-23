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

  /**
   * @brief Constructs the required parameters for the oversample remap kernel.
   *
   * @param oversampleFactor Oversample factor (number of pixels interpolated
   * per output pixel)
   * @param outStep Distance between final output pixels in terms of input pixel
   * coordinates e.g.
   * I ----- I ----- I -> input
   * O   O   O   O   O -> outStep of 0.5
   * @param outOffset Starting output pixel point in terms of input pixel
   * coordinates. Origin is defined as the first input pixel i.e. index 0,0
   */
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

// These rotation parameters are applied to the oversampled pixels
template <typename Tcalc> struct RotationParams {
  static_assert(std::is_floating_point<Tcalc>::value,
                "Tcalc must be float or double");
  Tcalc xCentre;
  Tcalc yCentre;
  Tcalc rotmat[2][2];
  bool used;

  __host__ RotationParams() { this->used = false; }

  __host__ RotationParams(Tcalc xCentre, Tcalc yCentre, Tcalc angleRadians) {
    this->xCentre = xCentre;
    this->yCentre = yCentre;
    // Compute matrix in double, since it's on host and 1-time anyway
    this->rotmat[0][0] = cos((double)angleRadians);
    this->rotmat[0][1] = -sin((double)angleRadians);
    this->rotmat[1][0] = sin((double)angleRadians);
    this->rotmat[1][1] = cos((double)angleRadians);
    this->used = true;
  }

  __host__ __device__ cuda_vec2_t<Tcalc> rotate(const Tcalc x,
                                                const Tcalc y) const {
    cuda_vec2_t<Tcalc> out;
    // find the actual coordinate with respect to the centre, not just the pixel
    // coordinate
    Tcalc xc = x - xCentre;
    Tcalc yc = y - yCentre;
    // rotate around this centre
    out.x = rotmat[0][0] * xc + rotmat[0][1] * yc;
    out.y = rotmat[1][0] * xc + rotmat[1][1] * yc;
    // add the centre back to find the pixel coordinate used
    out.x += xCentre;
    out.y += yCentre;
    return out;
  }
};

// All coordinates use the input image coordinates
template <typename Tin, typename Tout, typename Tcalc = float,
          bool useSharedMem = true>
__global__ void oversampleBilerpAndCombineKernel(
    containers::Image<const Tin> in, containers::Image<Tout> out,
    const int2 oversampleFactor, // assumed odd
    const OversampleKernelParams<Tcalc> params,
    // NOTE: Surprisingly, ncu says computing numOutBlocks stalls a lot, and in
    // fact just passing this in directly speeds up about 4%
    const int2 numOutBlocks, const RotationParams<Tcalc> rotParams
    // NOTE: the mere existence of the parameters appears to slow down the
    // kernel, even if it is not used; maybe templatize this
) {
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
        // TODO: shmem method is harder to reason about for rotated points
        // implement later for rotated points; for now this is kept here
        // since it's much faster when we can use single precision

        // Compute the starting tile position and dimensions
        Tcalc tileStartY =
            oversampBlkStartYIdx * params.overSampStepY + params.overSampStartY;
        Tcalc tileStartX =
            oversampBlkStartXIdx * params.overSampStepX + params.overSampStartX;

        // NOTE: this is technically fixed
        int tilewidth =
            (oversampleFactor.x * blockDim.x) * params.overSampStepX +
            3; // +1 both sides for safety
        int tileheight =
            (oversampleFactor.y * blockDim.y) * params.overSampStepY +
            3; // +1 both sides for safety

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
            Tin topLeft = stile.at(iy, ix);
            Tin topRight = stile.at(iy, ix + 1);
            Tin botLeft = stile.at(iy + 1, ix);
            Tin botRight = stile.at(iy + 1, ix + 1);

            Tcalc interpolated = bilinearInterpolate<Tin, Tcalc>(
                topLeft, topRight, botLeft, botRight, sx, sy);

            value += interpolated;
          }
        }
      } else {
        for (int oy = 0; oy < oversampleFactor.y; ++oy) {
          int y = threadIdx.y * oversampleFactor.y + oy;
          for (int ox = 0; ox < oversampleFactor.x; ++ox) {
            int x = threadIdx.x * oversampleFactor.x + ox;
            // Compute interpolated value directly
            Tcalc sy = (oversampBlkStartYIdx + y) * params.overSampStepY +
                       params.overSampStartY;
            Tcalc sx = (oversampBlkStartXIdx + x) * params.overSampStepX +
                       params.overSampStartX;
            // sx, sy are the coordinates to bilerp
            if (rotParams.used) {
              // rotate the coordinates
              cuda_vec2_t<Tcalc> rotated = rotParams.rotate(sx, sy);
              sx = rotated.x;
              sy = rotated.y;
            }

            // we get the nearest integer index to read from
            int iy = (int)cuda::std::floor(sy);
            int ix = (int)cuda::std::floor(sx);

            // Extract the 4 corners of data
            // Here is where you would include logic to handle pixel reading
            // For now we just read simply and default to 0 if it doesn't exist
            Tin topLeft = in.atWithDefault(iy, ix);
            Tin topRight = in.atWithDefault(iy, ix + 1);
            Tin botLeft = in.atWithDefault(iy + 1, ix);
            Tin botRight = in.atWithDefault(iy + 1, ix + 1);

            Tcalc interpolated = bilinearInterpolate<Tin, Tcalc>(
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
                                const dim3 tpb, const Tcalc angleRadians = 0,
                                cudaStream_t stream = 0) {
  // Throw if not odd oversample
  if (oversampleFactor.x % 2 == 0 || oversampleFactor.y % 2 == 0) {
    throw std::runtime_error("oversampleFactor must be odd in both dimensions");
  }

  // Compute oversample kernel parameters externally
  OversampleKernelParams<Tcalc> params(oversampleFactor, outStep, outOffset);

  // Compute rotation parameters
  RotationParams<Tcalc> rotParams; // defaults to unused
  if (angleRadians != 0) {
    Tcalc xCentre = in.width / 2 - (in.width % 2 == 0 ? 0.5 : 0);
    Tcalc yCentre = in.height / 2 - (in.height % 2 == 0 ? 0.5 : 0);
    printf("centre is at %f, %f\n", xCentre, yCentre);
    rotParams = RotationParams<Tcalc>(xCentre, yCentre, angleRadians);
    printf("%12.8f %12.8f\n%12.8f %12.8f\n", rotParams.rotmat[0][0],
           rotParams.rotmat[0][1], rotParams.rotmat[1][0],
           rotParams.rotmat[1][1]);
  }

  // Compute number of blocks to cover the output
  dim3 outblks(justEnoughBlocks(tpb.x, (unsigned int)out.width),
               justEnoughBlocks(tpb.y, (unsigned int)out.height));
  size_t shmem = 0;

  // DEPRECATED. LATER CHANGE WHEN NEEDED FOR INPUT TILE SHMEM
  if constexpr (useSharedMem) {
    // NOTE: this is technically fixed
    // TODO: make it so we don't calculate outside and inside
    int tilewidth = (oversampleFactor.x * tpb.x) * params.overSampStepX +
                    3; // +1 both sides for safety
    int tileheight = (oversampleFactor.y * tpb.y) * params.overSampStepY +
                     3; // +1 both sides for safety
    shmem = tilewidth * tileheight * sizeof(Tin);
    printf("shmem tile is %dx%d, %zu bytes\n", tilewidth, tileheight, shmem);
  }

  // Possibly adjust the number of blocks of grid to not cover output
  dim3 blks(outblks.x,
            outblks.y); // early testing shows possibly 1-2% speedup if we use
                        // 1/2 total blocks for example, probably only relevant
                        // when number of blocks is large

  // printf("Using shared mem %zu bytes\n", shmem);
  oversampleBilerpAndCombineKernel<Tin, Tout, Tcalc, useSharedMem>
      <<<blks, tpb, shmem, stream>>>(in, out, oversampleFactor, params,
                                     int2{(int)outblks.x, (int)outblks.y},
                                     rotParams);
}

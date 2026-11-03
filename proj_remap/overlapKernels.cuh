#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include <cuda/std/limits>

#include "block_and_grid_sizing.cuh"
#include "containers/image.cuh"
#include "extra_type_traits.cuh"
#include "sharedmem.cuh"

/** @brief Output-driven area-overlap remapping in input-pixel coordinates. */
namespace overlap {

/**
 * @brief Affine map from output-pixel indices to input-pixel indices.
 *
 * A pixel centre is at its integer index; its box extends 0.5 on either side.
 * The two axes describe the change in input coordinates for a one-pixel step
 * along each output axis. They can encode rotation as well as pitch changes.
 *
 * In homogeneous coordinates, the stored vectors are the matrix columns:
 * @code
 * [inputX]   [xAxis.x  yAxis.x  origin.x] [outputX]
 * [inputY] = [xAxis.y  yAxis.y  origin.y] [outputY]
 * [  1   ]   [   0         0        1   ] [   1   ]
 * @endcode
 *
 * @tparam T Floating-point coordinate type (float or double).
 */
template <typename T> struct OutputToInput {
  cuda_vec2_t<T> xAxis;  ///< Input-space displacement for +1 output X.
  cuda_vec2_t<T> yAxis;  ///< Input-space displacement for +1 output Y.
  cuda_vec2_t<T> origin; ///< Input-space location of output centre (0, 0).

  /**
   * @brief Locate an output-space point in input-pixel coordinates.
   * @param x Output-space X, measured in output pixels.
   * @param y Output-space Y, measured in output pixels.
   * @return The corresponding location in input-pixel coordinates.
   */
  __host__ __device__ cuda_vec2_t<T> operator()(T x, T y) const {
    return {xAxis.x * x + yAxis.x * y + origin.x,
            xAxis.y * x + yAxis.y * y + origin.y};
  }

  /**
   * @brief Convert an input-space intersection area to output-pixel-square units.
   * @return Reciprocal absolute determinant of the output-to-input transform.
   */
  __host__ __device__ T areaScale() const {
    return T(1) / fabs(xAxis.x * yAxis.y - xAxis.y * yAxis.x);
  }
};

/**
 * @brief One input pixel's contribution to an output pixel.
 * @tparam T Floating-point accumulation type.
 */
template <typename T> struct Contribution {
  T weight; ///< Area-based weight, in output-pixel-square units.
  T value;  ///< Input value multiplied by that weight.
};

/**
 * @brief Intersect a transformed output pixel with one axis-aligned input box.
 *
 * Clips the four-corner output polygon against the input pixel's left, right,
 * bottom, and top boundaries. The resulting polygon has at most eight vertices.
 * The signed shoelace sum gives its area without raster sampling.
 *
 * @tparam T Floating-point coordinate type.
 * @param quad Output pixel's corners in input coordinates, in boundary order.
 * @param x Input pixel's integer X centre.
 * @param y Input pixel's integer Y centre.
 * @return Intersection area in input-pixel-square units.
 */
template <typename T>
__device__ T intersectionArea(const cuda_vec2_t<T> (&quad)[4], int x, int y) {
  cuda_vec2_t<T> vertices[12], next[12];
  for (int i = 0; i < 4; ++i)
    vertices[i] = quad[i];
  int count = 4;

  // Clip in order: left, right, bottom, top of the candidate input pixel.
  for (int side = 0; side < 4 && count; ++side) {
    const bool xAxis = side < 2;
    const T bound = (side % 2 ? T(0.5) : T(-0.5)) + (xAxis ? x : y);
    const bool greater = side % 2 == 0;
    int nextCount = 0;

    // Walk every polygon edge, retaining inside vertices and new crossings.
    for (int i = 0; i < count; ++i) {
      const cuda_vec2_t<T> a = vertices[i];
      const cuda_vec2_t<T> b = vertices[(i + 1) % count];
      const T av = xAxis ? a.x : a.y;
      const T bv = xAxis ? b.x : b.y;
      const bool aInside = greater ? av >= bound : av <= bound;
      const bool bInside = greater ? bv >= bound : bv <= bound;
      // Endpoints on different sides imply one boundary crossing.
      if (aInside != bInside) {
        const T t = (bound - av) / (bv - av);
        next[nextCount++] = {a.x + t * (b.x - a.x),
                             a.y + t * (b.y - a.y)};
      }
      if (bInside)
        next[nextCount++] = b;
    }
    // Feed the clipped boundary into the next box-boundary pass.
    count = nextCount;
    for (int i = 0; i < count; ++i)
      vertices[i] = next[i];
  }

  // Directed cross products sum to twice the signed polygon area.
  T sum = 0;
  for (int i = 0; i < count; ++i) {
    const cuda_vec2_t<T> a = vertices[i], b = vertices[(i + 1) % count];
    sum += a.x * b.y - b.x * a.y;
  }
  return fabs(sum) * T(0.5);
}

/**
 * @brief Default accumulation policy: area-weighted input value.
 *
 * Samples equal to the type's maximum are invalid and contribute nothing.
 * Replace this functor to change validity or weighting without changing the
 * output traversal or geometric candidate search.
 *
 * @tparam T Floating-point accumulation type.
 */
template <typename T> struct AreaAccumulator {
  /**
   * @brief Calculate the weight and weighted value for one candidate pixel.
   * @tparam Tin Input pixel value type.
   * @param quad Output-pixel polygon in input coordinates.
   * @param x Candidate input pixel's X index.
   * @param y Candidate input pixel's Y index.
   * @param sample Value read from the input image.
   * @param areaScale Input-to-output area conversion.
   * @return Weight and weighted value, or zeros for an invalid sample.
   */
  template <typename Tin>
  __device__ Contribution<T>
  operator()(const cuda_vec2_t<T> (&quad)[4], int x, int y, Tin sample,
             T areaScale) const {
    if (sample == cuda::std::numeric_limits<Tin>::max())
      return {0, 0};
    // A whole output pixel has area 1 in output-pixel-square units.
    const T weight = intersectionArea<T>(quad, x, y) * areaScale;
    return {weight, weight * static_cast<T>(sample)};
  }
};

/**
 * @brief Gather one Z worker's share of an output pixel's input candidates.
 *
 * All workers for the same output pixel visit every candidate input row, but
 * worker 0 takes input columns 0, worker 1 takes column 1, and so on, stepping
 * by workerCount. With one worker this is the original serial gather.
 *
 * @param in Read-only input image.
 * @param transform Output-to-input affine transform.
 * @param ox Output pixel X.
 * @param oy Output pixel Y.
 * @param accumulate Per-candidate contribution policy.
 * @param worker This output pixel's Z worker index.
 * @param workerCount Number of Z workers per output pixel.
 * @return This worker's partial weight and weighted value.
 */
template <typename Tin, typename T, typename Accumulator>
__device__ Contribution<T>
gatherOverlap(containers::Image<const Tin> in, OutputToInput<T> transform,
              int ox, int oy, Accumulator accumulate, int worker,
              int workerCount) {
  // Map the four output-box corners into input-pixel coordinates.
  cuda_vec2_t<T> quad[4] = {
      transform(ox - T(0.5), oy - T(0.5)),
      transform(ox + T(0.5), oy - T(0.5)),
      transform(ox + T(0.5), oy + T(0.5)),
      transform(ox - T(0.5), oy + T(0.5)),
  };
  // An input box can intersect the quad only near its bounding rectangle.
  T minX = quad[0].x, maxX = minX, minY = quad[0].y, maxY = minY;
  for (int i = 1; i < 4; ++i) {
    minX = fmin(minX, quad[i].x);
    maxX = fmax(maxX, quad[i].x);
    minY = fmin(minY, quad[i].y);
    maxY = fmax(maxY, quad[i].y);
  }

  // Expand by half an input pixel and stay within the input Image.
  const int firstX = max(0, static_cast<int>(ceil(minX - T(0.5))));
  const int lastX = min(in.width - 1, static_cast<int>(floor(maxX + T(0.5))));
  const int firstY = max(0, static_cast<int>(ceil(minY - T(0.5))));
  const int lastY = min(in.height - 1, static_cast<int>(floor(maxY + T(0.5))));
  const T areaScale = transform.areaScale();
  Contribution<T> total{0, 0};
  for (int iy = firstY; iy <= lastY; ++iy)
    for (int ix = firstX + worker; ix <= lastX; ix += workerCount) {
      const auto part = accumulate(quad, ix, iy, in.at(iy, ix), areaScale);
      total.weight += part.weight;
      total.value += part.value;
    }
  return total;
}

/**
 * @brief Gather area contributions by striding over both output-grid axes.
 *
 * Each thread owns its output pixel's sum and weight, so writes need no atomics.
 * No shared memory is used: candidates are read directly from the input Image.
 *
 * @tparam Tin Input pixel type.
 * @tparam Tout Output pixel type.
 * @tparam T Coordinate and accumulation type.
 * @tparam Accumulator Device-callable policy returning Contribution-like fields.
 * @param in Read-only input Image, including its dimensions and row stride.
 * @param out Output Image, including its dimensions and row stride.
 * @param transform Output-to-input coordinate mapping.
 * @param minWeight Strict minimum weight for a valid output.
 * @param accumulate Functor called for each candidate input pixel.
 */
template <typename Tin, typename Tout, typename T, typename Accumulator>
__global__ void overlapKernel(containers::Image<const Tin> in,
                              containers::Image<Tout> out,
                              OutputToInput<T> transform, T minWeight,
                              Accumulator accumulate) {
  static_assert(std::is_floating_point<T>::value,
                "overlap coordinates must use float or double");
  // Every thread advances through output columns and rows independently.
  const int strideX = blockDim.x * gridDim.x;
  const int strideY = blockDim.y * gridDim.y;
  for (int oy = blockIdx.y * blockDim.y + threadIdx.y; oy < out.height;
       oy += strideY) {
    for (int ox = blockIdx.x * blockDim.x + threadIdx.x; ox < out.width;
         ox += strideX) {
      const auto total = gatherOverlap(in, transform, ox, oy, accumulate, 0, 1);
      // The output is valid only when its accumulated weight clears the threshold.
      out.at(oy, ox) = total.weight > minWeight
                           ? static_cast<Tout>(total.value / total.weight)
                           : cuda::std::numeric_limits<Tout>::max();
    }
  }
}

/**
 * @brief Split each output pixel's input columns between blockDim.z workers.
 *
 * In a 32x2x4 block there are 64 output pixels and four Z workers per pixel.
 * Each worker gathers different input columns for the same (x, y). Two arrays
 * in shared memory hold the partial weights and weighted values:
 *
 * Let P = blockDim.x * blockDim.y (64 in the example). The first P*blockDim.z
 * elements hold weights; the next P*blockDim.z elements hold weighted values.
 * The location of an output pixel within one Z layer is
 *   pixelSlot = threadIdx.y * blockDim.x + threadIdx.x.
 * Each Z worker owns a DIFFERENT weight slot and a DIFFERENT value slot:
 *   slot = threadIdx.z * P + pixelSlot.
 * After workers store their register sums, Z==0 reads every Z slot for its
 * pixel. For 32x2x4, it sums pixelSlot, 64+pixelSlot, 128+pixelSlot, and
 * 192+pixelSlot from each array, then writes the one output pixel. Workers
 * never share a write address, so no atomics are needed.
 *
 * The X/Y grid-stride loops advance by whole tiles, identically for EVERY
 * thread in the block. Even threads whose X/Y falls outside the image reach
 * all barriers; they contribute zero. The final barrier prevents the next
 * tile from overwriting shared slots before Z==0 has read the previous tile.
 *
 * @param in Read-only input Image.
 * @param out Output Image.
 * @param transform Output-to-input coordinate mapping.
 * @param minWeight Strict minimum area weight for a valid output pixel.
 * @param accumulate Functor returning a candidate's weight and weighted value.
 */
template <typename Tin, typename Tout, typename T, typename Accumulator>
__global__ void overlapSharedPartialsKernel(containers::Image<const Tin> in,
                                            containers::Image<Tout> out,
                                            OutputToInput<T> transform,
                                            T minWeight,
                                            Accumulator accumulate) {
  static_assert(std::is_floating_point<T>::value,
                "overlap coordinates must use float or double");
  SharedMemory<T> sharedMemory;
  T *shared = sharedMemory.getPointer();
  const int pixelsPerBlock = blockDim.x * blockDim.y;
  const int pixelSlot = threadIdx.y * blockDim.x + threadIdx.x;
  const int slots = pixelsPerBlock * blockDim.z;
  T *sharedWeights = shared;
  T *sharedValues = shared + slots;

  const int strideX = gridDim.x * blockDim.x;
  const int strideY = gridDim.y * blockDim.y;
  for (int baseY = blockIdx.y * blockDim.y; baseY < out.height;
       baseY += strideY) {
    for (int baseX = blockIdx.x * blockDim.x; baseX < out.width;
         baseX += strideX) {
      const int ox = baseX + threadIdx.x, oy = baseY + threadIdx.y;
      const bool active = ox < out.width && oy < out.height;

      const Contribution<T> part =
          active ? gatherOverlap(in, transform, ox, oy, accumulate,
                                 threadIdx.z, blockDim.z)
                 : Contribution<T>{0, 0};
      // Every worker has private slots, including on incomplete edge tiles.
      const int slot = threadIdx.z * pixelsPerBlock + pixelSlot;
      sharedWeights[slot] = part.weight;
      sharedValues[slot] = part.value;
      // Every worker must store its partials before Z==0 can read them.
      __syncthreads();

      if (threadIdx.z == 0 && active) {
        T weight = 0, value = 0;
        for (int z = 0; z < blockDim.z; ++z) {
          const int slot = z * pixelsPerBlock + pixelSlot;
          weight += sharedWeights[slot];
          value += sharedValues[slot];
        }
        // Exactly one thread per output pixel writes to global memory.
        out.at(oy, ox) = weight > minWeight
                             ? static_cast<Tout>(value / weight)
                             : cuda::std::numeric_limits<Tout>::max();
      }
      // Keep the current tile's shared contents intact until Z==0 finishes.
      __syncthreads();
    }
  }
}

/**
 * @brief Launch the output-driven overlap remap asynchronously.
 *
 * @tparam Tin Input pixel type.
 * @tparam Tout Output pixel type.
 * @tparam T Floating-point coordinate and accumulation type.
 * @tparam Accumulator Policy for calculating one candidate's contribution.
 * @param in Read-only input Image; max-valued samples are invalid by default.
 * @param out Output Image; invalid pixels receive the type's maximum.
 * @param transform Invertible output-to-input affine mapping.
 * @param minWeight Strict minimum output-area weight; zero accepts any overlap.
 * @param accumulator Custom policy; defaults to area-weighted averaging.
 * @param blocks Optional grid override; zero on either axis uses just enough
 * blocks to cover that output dimension.
 * @param threads Threads per block along X and Y; defaults to 32 by 4.
 * @param stream CUDA stream; callers synchronize before reading results.
 */
template <typename Tin, typename Tout, typename T = double,
          typename Accumulator = AreaAccumulator<T>>
void remapOverlap(containers::Image<const Tin> in, containers::Image<Tout> out,
           OutputToInput<T> transform, T minWeight = 0,
           Accumulator accumulator = {}, dim3 blocks = dim3(0, 0),
           dim3 threads = dim3(32, 4), cudaStream_t stream = 0) {
  // A singular mapping cannot convert area between the two pixel grids.
  if (transform.xAxis.x * transform.yAxis.y ==
      transform.xAxis.y * transform.yAxis.x)
    throw std::invalid_argument("output-to-input transform must be invertible");
  if (minWeight < T(0))
    throw std::invalid_argument("minimum weight must be non-negative");
  if (out.width == 0 || out.height == 0)
    return;
  // By default cover the output once; smaller explicit grids still stride.
  const dim3 launchBlocks(
      blocks.x ? blocks.x
               : justEnoughBlocks(threads.x, static_cast<unsigned int>(out.width)),
      blocks.y ? blocks.y
               : justEnoughBlocks(threads.y, static_cast<unsigned int>(out.height)));
  overlapKernel<<<launchBlocks, threads, 0, stream>>>(
      in, out, transform, minWeight, accumulator);
}

/**
 * @brief Launch the shared-partials overlap gather with Z workers per output.
 *
 * The default 32x2x4 block handles 64 output pixels at a time, with four
 * workers per pixel. Specify an undersized X/Y grid to exercise grid-striding.
 * Two arrays of 64 * blockDim.z T values hold private per-worker partials.
 *
 * @tparam Tin Input pixel type.
 * @tparam Tout Output pixel type.
 * @tparam T Floating-point coordinate and accumulation type.
 * @tparam Accumulator Candidate contribution policy.
 * @param in Read-only input image.
 * @param out Output image.
 * @param transform Invertible output-to-input mapping.
 * @param minWeight Strict minimum output-area weight.
 * @param accumulator Contribution policy.
 * @param blocks Optional X/Y grid size, automatically sized when zero.
 * @param threads Block dimensions (X pixels, Y pixels, Z workers).
 * @param stream CUDA stream.
 */
template <typename Tin, typename Tout, typename T = double,
          typename Accumulator = AreaAccumulator<T>>
void remapOverlapSharedPartials(containers::Image<const Tin> in,
                                containers::Image<Tout> out,
                                OutputToInput<T> transform, T minWeight = 0,
                                Accumulator accumulator = {},
                                dim3 blocks = dim3(0, 0),
                                dim3 threads = dim3(32, 2, 4),
                                cudaStream_t stream = 0) {
  if (transform.xAxis.x * transform.yAxis.y ==
      transform.xAxis.y * transform.yAxis.x)
    throw std::invalid_argument("output-to-input transform must be invertible");
  if (minWeight < T(0))
    throw std::invalid_argument("minimum weight must be non-negative");
  if (out.width == 0 || out.height == 0)
    return;
  const dim3 launchBlocks(
      blocks.x ? blocks.x
               : justEnoughBlocks(threads.x, static_cast<unsigned int>(out.width)),
      blocks.y ? blocks.y
               : justEnoughBlocks(threads.y, static_cast<unsigned int>(out.height)));
  const size_t sharedBytes = 2 * threads.x * threads.y * threads.z * sizeof(T);
  overlapSharedPartialsKernel<<<launchBlocks, threads, sharedBytes, stream>>>(
      in, out, transform, minWeight, accumulator);
}

}

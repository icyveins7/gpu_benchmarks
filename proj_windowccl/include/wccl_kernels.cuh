#pragma once

#include <cstdint>
#include <stdexcept>
#include <thrust/device_vector.h>

#include "accessors.cuh"
#include "sharedmem.cuh"

namespace wccl {

template <typename T> struct DeviceImage {
  T *data; // non-owning, but we leave as raw pointer for CUDA stuff
  unsigned int height;
  unsigned int width;
  unsigned int yOffset = 0; // only used when we are looking
  unsigned int xOffset = 0; // at a tile inside kernels

  DeviceImage() = default;
  DeviceImage(thrust::device_vector<T> &_data, int _height, int _width)
      : data(_data.data().get()), height(_height), width(_width) {
    if (data == nullptr) {
      throw std::runtime_error("data is null");
    }
    if (height * width != _data.size())
      throw std::runtime_error("data size is not equal to height * width");
  }

  /**
   * @brief Returns the number of elements in the image
   */
  __host__ __device__ size_t size() { return height * width; }

  /**
   * @brief Returns the flattened 1D index from the row and column indices.
   */
  __host__ __device__ unsigned int flattenedIndex(unsigned int row,
                                                  unsigned int col) {
    return row * width + col;
  }
  /**
   * @brief Returns the global flattened 1D index from the row and column
   * indices. This accounts for the offsets of the current tile.
   */
  __host__ __device__ unsigned int globalIndex(unsigned int row,
                                               unsigned int col) {
    return (row + yOffset) * width + (col + xOffset);
  }

  /**
   * @brief Range-checked access via row/col indices.
   */
  __host__ __device__ T &at(unsigned int row, unsigned int col) {
    if (row >= height || col >= width)
      return data[flattenedIndex(height, width)];
    else
      return cuda::std::numeric_limits<T>::max();
  }

  /**
   * @brief (Const) Range-checked access via row/col indices.
   */
  __host__ __device__ const T &at(unsigned int row, unsigned int col) const {
    return (const T &)at(row, col);
  }
};

template <typename T>
__device__ T find(const DeviceImage<T> &tile, const int row, const int col) {
  int idx = tile.flattenedIndex(row, col);
  // Return directly if inactive i.e. -1 -> -1
  if (tile.data[idx] < 0)
    return tile.data[idx];

  // Walk entire thing to find root
  while (tile.data[idx] != idx) {
    idx = tile.data[idx];
  }
  return idx;
}

template <typename T>
__device__ void unite(const DeviceImage<T> &tile_in, DeviceImage<T> &tile_out,
                      const int2 windowDist) {
  for (int ty = threadIdx.y; ty < tile_in.height; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < tile_in.width; tx += blockDim.x) {
      T root = find(tile_in, ty, tx);
      if (root < 0)
        continue;

      // Loop over the surrounding window
      for (int wy = ty - windowDist.y; wy <= ty + windowDist.y; wy++) {
        for (int wx = tx - windowDist.x; wx <= tx + windowDist.x; wx++) {
          if (wy >= 0 && wy < tile_in.height && wx >= 0 && wx < tile_in.width) {
            T windowRoot = find(tile_in, wy, wx);
            if (windowRoot < 0)
              continue;

            // Take min root
            root = root < windowRoot ? root : windowRoot;
          }
        }
      }

      // Write to output tile
      tile_out.at(ty, tx) = root;
    }
  }
}

template <typename T>
__global__ void connect_kernel(const DeviceImage<uint8_t> &input,
                               DeviceImage<T> &mapping, const int2 tileDims,
                               const int2 windowDist) {
  static_assert(std::is_signed_v<T>, "mapping type T must be signed");

  // Define the tile for this block
  int tileXstart = blockIdx.x * tileDims.x;
  int tileYstart = blockIdx.y * tileDims.y;

  // Read entire tile and set shared memory values
  SharedMemory<T> smem;
  // Encapsulate shared memory tile for ergonomics
  DeviceImage<T> s_tile1{smem.getPointer(), tileDims.y, tileDims.x, tileYstart,
                         tileXstart};
  // We have 2 workspaces
  DeviceImage<T> s_tile2{&s_tile1.data[s_tile1.size()], tileDims.y, tileDims.x,
                         tileYstart, tileXstart};

  for (int ty = threadIdx.y; ty < tileDims.y; ty += blockDim.y) {
    // Define global index for reads (row)
    int y = tileYstart + ty;
    for (int tx = threadIdx.x; tx < tileDims.x; tx += blockDim.x) {
      // Define global index for reads (col)
      int x = tileXstart + tx;
      // Global read; if inactive, set to -1, otherwise set to flattened local
      // index (for now)
      s_tile1.at(ty, tx) =
          input.at(y, x) > 0 ? s_tile1.flattenedIndex(ty, tx) : -1;
    }
  }
  __syncthreads();

  // Perform first unites
  unite(s_tile1, s_tile2, windowDist);
}

} // namespace wccl

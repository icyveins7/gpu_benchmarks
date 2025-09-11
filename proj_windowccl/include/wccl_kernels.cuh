#pragma once

#include <cstdint>
#include <stdexcept>
#include <thrust/device_vector.h>

#include "accessors.cuh"
#include "sharedmem.cuh"

namespace wccl {

// NOTE: it is important that when using this as inputs to a kernel, a
// reference/pointer is not used, since it is merely a container for a copy of
// the pointers and sizes i.e. just a handy substitute for using 4 separate
// arguments in the kernel inputs. using a reference/pointer will undoubtedly
// break the kernel.
template <typename T> struct DeviceImage {
  T *data; // non-owning, but we leave as raw pointer for CUDA stuff
  unsigned int height;
  unsigned int width;
  unsigned int yOffset = 0; // only used when we are looking
  unsigned int xOffset = 0; // at a tile inside kernels

  DeviceImage() = default;
  __device__ DeviceImage(T *_data, int _height, int _width, int _yOffset = 0,
                         int _xOffset = 0)
      : data(_data), height(_height), width(_width), yOffset(_yOffset),
        xOffset(_xOffset) {}
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
  __host__ __device__ size_t size() const { return height * width; }

  /**
   * @brief Returns the flattened 1D index from the row and column indices.
   */
  __host__ __device__ unsigned int flattenedIndex(unsigned int row,
                                                  unsigned int col) const {
    return row * width + col;
  }
  /**
   * @brief Returns the global flattened 1D index from the row and column
   * indices. This accounts for the offsets of the current tile.
   */
  __host__ __device__ unsigned int globalIndex(unsigned int row,
                                               unsigned int col) const {
    return (row + yOffset) * width + (col + xOffset);
  }

  /**
   * @brief Range-checked getter via row/col indices.
   */
  __host__ __device__ T get(const unsigned int row,
                            const unsigned int col) const {
    if (row >= height || col >= width)
      return cuda::std::numeric_limits<T>::max();
    return data[flattenedIndex(row, col)];
  }

  /**
   * @brief Range-checked setter via row/col indices. No-op if out of bounds.
   */
  __host__ __device__ void set(const unsigned int row, const unsigned int col,
                               const T val) {
    if (row < height && col < width)
      data[flattenedIndex(row, col)] = val;
  }
};

template <typename T>
__device__ T find(const DeviceImage<T> &tile, const unsigned int row,
                  const unsigned int col) {
  unsigned int idx = tile.flattenedIndex(row, col);
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

      if (root >= 0) {
        // Loop over the surrounding window
        for (int wy = ty - windowDist.y; wy <= ty + windowDist.y; wy++) {
          for (int wx = tx - windowDist.x; wx <= tx + windowDist.x; wx++) {
            if (wy >= 0 && wy < tile_in.height && wx >= 0 &&
                wx < tile_in.width) {
              T windowRoot = find(tile_in, wy, wx);
              if (windowRoot < 0)
                continue;

              // Take min root
              root = root < windowRoot ? root : windowRoot;
            }
          }
        }
      }

      // Write to output tile
      printf("Wrote root %d to (%d, %d)\n", root, ty, tx);
      tile_out.set(ty, tx, root);
    }
  }
}

template <typename T>
__global__ void connect_kernel(const DeviceImage<uint8_t> input,
                               DeviceImage<T> mapping, const int2 tileDims,
                               const int2 windowDist) {
  static_assert(std::is_signed_v<T>, "mapping type T must be signed");

  // Define the tile for this block
  const int tileXstart = blockIdx.x * tileDims.x;
  const int tileYstart = blockIdx.y * tileDims.y;
  // Tile may not be fully occupied at the ends
  const int2 blockTileDims{tileXstart + tileDims.x < (int)input.width
                               ? tileDims.x
                               : (int)input.width - tileXstart,
                           tileYstart + tileDims.y < (int)input.height
                               ? tileDims.y
                               : (int)input.height - tileYstart};

  // Read entire tile and set shared memory values
  SharedMemory<T> smem;
  // Encapsulate shared memory tile for ergonomics
  // We have 2 workspaces
  DeviceImage<T> s_tiles[2] = {
      DeviceImage<T>(smem.getPointer(), blockTileDims.y, blockTileDims.x,
                     tileYstart, tileXstart),
      DeviceImage<T>(&s_tiles[0].data[s_tiles[0].size()], blockTileDims.y,
                     blockTileDims.x, tileYstart, tileXstart)};

  for (int ty = threadIdx.y; ty < tileDims.y; ty += blockDim.y) {
    // Define global index for reads (row)
    int y = tileYstart + ty;
    for (int tx = threadIdx.x; tx < tileDims.x; tx += blockDim.x) {
      // Define global index for reads (col)
      int x = tileXstart + tx;

      // Global read; if inactive (or out of bounds), set to -1, otherwise
      // set to flattened local index (for now)
      if (y < (int)input.height && x < (int)input.width) {
        const uint8_t site = input.get(y, x);
        const T initVal = site > 0 ? s_tiles[0].flattenedIndex(ty, tx) : -1;

        printf("tile(%d,%d)<-input.get(%d,%d): %d\n", ty, tx, y, x, initVal);
        s_tiles[0].set(ty, tx, initVal);
      } else {
        // printf("Out of bounds at (%d, %d) in s_tiles[0]\n", y, x);
        s_tiles[0].set(ty, tx, -1);
      }
    }
  }
  __syncthreads();

  // Perform unites
  const int numUnites = 1;
  for (int i = 0; i < numUnites; ++i) {
    unite(s_tiles[i % 2], s_tiles[(i + 1) % 2], windowDist);
    __syncthreads();
  }

  // TODO: for now, just write this out to global
  for (int ty = threadIdx.y; ty < tileDims.y; ty += blockDim.y) {
    int y = tileYstart + ty;
    if (y >= (int)mapping.height)
      continue;
    for (int tx = threadIdx.x; tx < tileDims.x; tx += blockDim.x) {
      int x = tileXstart + tx;
      if (x >= (int)mapping.width)
        continue;

      // printf("setting mapping.at(%d, %d)\n", y, x);
      mapping.set(y, x, s_tiles[numUnites % 2].get(ty, tx));
    }
  }
}

} // namespace wccl

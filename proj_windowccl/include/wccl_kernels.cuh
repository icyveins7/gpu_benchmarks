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

  __host__ __device__ DeviceImage(T *_data, int _height, int _width)
      : data(_data), height(_height), width(_width) {}
  /**
   * @brief Specifically for host side code, to instantiate a container before
   * sending it into the kernel.
   */
  __host__ DeviceImage(thrust::device_vector<T> &_data, int _height, int _width)
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
   * @brief Range-checked getter via row/col indices.
   */
  __host__ __device__ T get(const unsigned int row,
                            const unsigned int col) const {
    if (row >= height || col >= width)
      return T(); // use default-constructed value?
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

  /**
   * @brief Range-checked getter via global row/col indices, removing tile
   * offsets automatically.
   *
   * @param gRow Global row index
   * @param gCol Global col index
   * @param tileXstart Tile column offset
   * @param tileYstart Tile row offset
   * @return Associated value from internal data
   */
  __host__ __device__ T get(const unsigned int gRow, const unsigned int gCol,
                            const int tileYstart, const int tileXstart) const {
    const int row = gRow - tileYstart;
    const int col = gCol - tileXstart;
    if (row >= 0 && col >= 0 && row < height && col < width)
      return data[flattenedIndex(row, col)];
    else
      return T();
  }
};

/**
 * @brief Specialization that should hold a pair type, representing row and
 * column indices.
 *
 * @tparam T Pair type, char2 or short2 or int2. Convention is {x, y} i.e. {col,
 * row}.
 */
template <typename Tcolrow>
struct DeviceIndexImage : public DeviceImage<Tcolrow> {
  static_assert(std::is_same_v<Tcolrow, char2> ||
                    std::is_same_v<Tcolrow, short2> ||
                    std::is_same_v<Tcolrow, int2>,
                "T must be either char2, short2 or int2");

  __device__ DeviceIndexImage() = default;
  __device__ DeviceIndexImage(Tcolrow *_data, int _height, int _width)
      : DeviceImage<Tcolrow>(_data, _height, _width) {}

  __device__ bool isInvalidAt(const unsigned int row,
                              const unsigned int col) const {
    Tcolrow element = this->data[this->flattenedIndex(row, col)];
    return element.x < 0 || element.y < 0;
  }

  __device__ void makeInvalidAt(const unsigned int row,
                                const unsigned int col) {
    Tcolrow element;
    element.x = -1;
    element.y = -1;
    this->data[this->flattenedIndex(row, col)] = element;
  }

  /**
   * @brief Returns True if the element at row, col contains (col, row).
   */
  __device__ bool isSelfRoot(const unsigned int row,
                             const unsigned int col) const {
    Tcolrow element = this->data[this->flattenedIndex(row, col)];
    return element.y == row && element.x == col;
  }

  /**
   * @brief Fills the element at row, col with (col, row).
   */
  __device__ void setInnateValueAt(const unsigned int row,
                                   const unsigned int col) {
    Tcolrow element{col, row};
    this->data[this->flattenedIndex(row, col)] = element;
  }

  /**
   * @brief Converts the internal row/col indices into a globally offset
   * flattened index.
   * @detail Primarily used when performing a final storage into global memory,
   * to account for tile offsets based on the current block. Specifically
   * ignores negative values (returns -1).
   *
   * @tparam U Type of global memory index
   * @param row The internal (local) row index to look up
   * @param col The internal (local) col index to look up
   * @param rowOffset Row offset of this image with respect to the global image
   * @param colOffset Col offset of this image with respect to the global image
   * @param globalWidth Global image width
   * @return
   */
  template <typename U>
  __device__ U readAndConvertToGlobalIndexFrom(const unsigned int row,
                                               const unsigned int col,
                                               const unsigned int rowOffset,
                                               const unsigned int colOffset,
                                               const unsigned int globalWidth) {
    if (this->isInvalidAt(row, col))
      return U(-1);
    // Calculate global row/col
    Tcolrow element = this->data[this->flattenedIndex(row, col)];
    Tcolrow globalColRow{element.x + colOffset, element.y + rowOffset};
    // Flatten the global row/col into 1D index
    return (U)(globalColRow.y * globalWidth + globalColRow.x);
  }
};

// ========================================================================
// ========================================================================
// ======================== KERNELS AND DEVICE FUNCS ======================
// ========================================================================
// ========================================================================

/**
 * @brief Local block finding, based on row/col indices.
 */
template <typename Tcolrow>
__device__ Tcolrow find(const DeviceIndexImage<Tcolrow> &tile, Tcolrow idx) {
  // Return directly if inactive i.e. -1 -> -1
  if (tile.isInvalidAt(idx.y, idx.x))
    return tile.get(idx.y, idx.x);

  // Walk entire thing to find root
  while (!tile.isSelfRoot(idx.y, idx.x)) {
    idx = tile.get(idx.y, idx.x);
  }
  return idx;
}

/**
 * @brief Global finder, may be highly warp divergent and will inevitably go
 * across global memory.
 */
template <typename Tmapping>
__device__ Tmapping *find(const DeviceImage<Tmapping> &img, Tmapping idx) {
  if (img.data[idx] < 0)
    return nullptr;

  while (img.data[idx] != idx) {
    idx = img.data[idx];
  }
  return &img.data[idx];
}

/**
 * @brief Does a pseudo-unite between two tiles.
 *
 * @details This does _NOT_ actually fully unite both sets, as this would
 * possibly require multiple threads to write to the same address. Specifically,
 * we try to prevent this by having each thread only rewrite to the address it
 * is currently working on. The downside of this is that 2 sets may not
 * immediately be united on one invocation, but the upside is that this method
 * does not require atomics (which were required when multiple threads could
 * update the same root address). Instead, this function is closer to a
 * 'neighbour' linkage. Hence, multiple iterations are required to fully unite
 * two sets. Very loosely, the number of iterations is dependent on the longest
 * 'chain'.
 *
 * @tparam Tcolrow Type of the col/row indices (see DeviceIndexImage)
 * @tparam Tmapping Type of the 1D flattened indices, used for comparisons
 * @param tile_in Input tile
 * @param tile_out Output tile
 * @param windowDist Window distance
 */
template <typename Tcolrow, typename Tmapping>
__device__ void pseudoUnite(const DeviceIndexImage<Tcolrow> &tile_in,
                            DeviceIndexImage<Tcolrow> &tile_out,
                            const int2 windowDist) {
  for (int ty = threadIdx.y; ty < tile_in.height; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < tile_in.width; tx += blockDim.x) {
      // Early exit (but still remember to copy the invalid value)
      if (tile_in.isInvalidAt(ty, tx)) {
        tile_out.makeInvalidAt(ty, tx);
        continue;
      }

      Tcolrow idx = Tcolrow{tx, ty};
      Tcolrow root = find(tile_in, idx);
      // Determine 1D index for comparisons
      Tmapping flatroot = tile_in.flattenedIndex(root.y, root.x);

      // Loop over the surrounding window
      for (int wy = ty - windowDist.y; wy <= ty + windowDist.y; wy++) {
        for (int wx = tx - windowDist.x; wx <= tx + windowDist.x; wx++) {
          if (wy >= 0 && wy < tile_in.height && wx >= 0 && wx < tile_in.width &&
              !tile_in.isInvalidAt(wy, wx) // If inactive, ignore
          ) {
            Tcolrow widx = Tcolrow{wx, wy};
            Tcolrow windowRoot = find(tile_in, widx);

            // Determine 1D index for comparisons
            Tmapping flatwindowRoot =
                tile_in.flattenedIndex(windowRoot.y, windowRoot.x);

            // Take min root
            root = flatroot < flatwindowRoot ? root : windowRoot;
            // remember to recompute the flatroot for comparisons!
            flatroot = tile_in.flattenedIndex(root.y, root.x);
            // printf("flatroot %d vs flatwindowRoot %d -> new root (%d, %d)\n",
            //        flatroot, flatwindowRoot, root.y, root.x);
          }
        }
      } // end loop over window

      // Write to output tile
      // printf("Wrote root (%d,%d) to (%d, %d)\n", root.y, root.x, ty, tx);
      tile_out.set(ty, tx, root);
    }
  }
}

template <typename Tmapping, typename Tcolrow>
__global__ void local_connect_kernel(const DeviceImage<uint8_t> input,
                                     DeviceImage<Tmapping> mapping,
                                     const int2 tileDims,
                                     const int2 windowDist) {
  static_assert(std::is_signed_v<Tmapping>, "mapping type T must be signed");

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
  SharedMemory<Tcolrow> smem;
  // Encapsulate shared memory tile for ergonomics
  // We have 2 workspaces
  DeviceIndexImage<Tcolrow> s_tiles[2] = {
      DeviceIndexImage<Tcolrow>(smem.getPointer(), blockTileDims.y,
                                blockTileDims.x),
      DeviceIndexImage<Tcolrow>(&s_tiles[0].data[s_tiles[0].size()],
                                blockTileDims.y, blockTileDims.x)};
  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    // Define global index for reads (row)
    int y = tileYstart + ty;
    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      // Define global index for reads (col)
      int x = tileXstart + tx;

      // Global read; if inactive (or out of bounds), set to -1, otherwise
      // set to flattened local index (for now)
      if (y < (int)input.height && x < (int)input.width) {
        const uint8_t site = input.get(y, x);
        if (site > 0)
          s_tiles[0].setInnateValueAt(ty, tx);
        else
          s_tiles[0].makeInvalidAt(ty, tx);

        // printf("blk(%d,%d) tile(%d,%d)<-input.get(%d,%d): (%d,%d)\n",
        //        blockIdx.x, blockIdx.y, ty, tx, y, x, s_tiles[0].get(ty,
        //        tx).y, s_tiles[0].get(ty, tx).x);
      } else {
        // printf("Out of bounds at (%d, %d) in s_tiles[0]\n", y, x);
        s_tiles[0].makeInvalidAt(ty, tx);
      }
    }
  }
  __syncthreads();

  // Perform unites
  int numUniteSteps = 0;
  constexpr int maxNumUnites = 100; // TODO: what's a good max iteration?
  for (; numUniteSteps < maxNumUnites; ++numUniteSteps) {
    pseudoUnite<Tcolrow, Tmapping>(s_tiles[numUniteSteps % 2],
                                   s_tiles[(numUniteSteps + 1) % 2],
                                   windowDist);
    // Detect workspace changes from input tile to output tile
    int threadChanges = 0;
    for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
      for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
        Tcolrow oldTileVal = s_tiles[numUniteSteps % 2].get(ty, tx);
        Tcolrow newTileVal = s_tiles[(numUniteSteps + 1) % 2].get(ty, tx);
        if (oldTileVal.x != newTileVal.x || oldTileVal.y != newTileVal.y) {
          threadChanges++;
        }
      }
    }
    // Sync block and check if any changes were effected
    int decision = __syncthreads_or(threadChanges);
    if (decision == 0)
      break;
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("numUniteSteps: %d\n", numUniteSteps);
    //   for (int i = 0; i < blockTileDims.y; ++i) {
    //     for (int j = 0; j < blockTileDims.x; ++j) {
    //       printf("(%d, %d): (%d, %d)\n", i, j,
    //              s_tiles[numUniteSteps % 2].get(i, j).y,
    //              s_tiles[numUniteSteps % 2].get(i, j).x);
    //     }
    //   }
  }

  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    int y = tileYstart + ty;
    if (y >= (int)mapping.height)
      continue;
    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      int x = tileXstart + tx;
      if (x >= (int)mapping.width)
        continue;

      Tmapping globalIdx =
          s_tiles[(numUniteSteps + 1) % 2]
              .template readAndConvertToGlobalIndexFrom<Tmapping>(
                  ty, tx, tileYstart, tileXstart, mapping.width);

      // TODO: add tile offset to global output
      // printf("setting mapping.at(%d, %d)\n", y, x);
      mapping.set(y, x, globalIdx);
    }
  }
}

template <typename Tmapping>
__global__ void inter_tile_neighbours_kernel(DeviceImage<Tmapping> mapping,
                                             const int2 tileDims,
                                             const int2 windowDist) {
  static_assert(std::is_signed_v<Tmapping>, "mapping type T must be signed");

  // Define the tile for this block
  const int tileXstart = blockIdx.x * tileDims.x;
  const int tileYstart = blockIdx.y * tileDims.y;
  // We then define an enlarged tile to include the border (this is what we will
  // load)
  const int borderTileXstart =
      tileXstart - windowDist.x < 0 ? 0 : tileXstart - windowDist.x;
  const int borderTileYstart =
      tileYstart - windowDist.y < 0 ? 0 : tileYstart - windowDist.y;
  // Tile may not be fully occupied at the ends
  const int2 blockTileDims{tileXstart + tileDims.x < (int)mapping.width
                               ? tileDims.x
                               : (int)mapping.width - tileXstart,
                           tileYstart + tileDims.y < (int)mapping.height
                               ? tileDims.y
                               : (int)mapping.height - tileYstart};

  const int2 borderBlockTileDims{
      borderTileXstart + tileDims.x + 2 * windowDist.x < (int)mapping.width
          ? tileDims.x + 2 * windowDist.x
          : (int)mapping.width - borderTileXstart,
      borderTileYstart + tileDims.y + 2 * windowDist.y < (int)mapping.height
          ? tileDims.y + 2 * windowDist.y
          : (int)mapping.height - borderTileYstart,
  };

  // Read tile with border into shared memory
  SharedMemory<Tmapping> smem;
  DeviceIndexImage<Tmapping> s_tiles(smem.getPointer(), borderBlockTileDims.y,
                                     borderBlockTileDims.x);

  blockRoiLoad(mapping.data, mapping.width, mapping.height, borderTileXstart,
               borderBlockTileDims.x, borderTileYstart, borderBlockTileDims.y,
               s_tiles.data);
  __syncthreads();

  // Now for every element, check for neighbours
  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      // We read elements based on their global indices instead of the local
      // ones, to track the internal tile WITHOUT the borders
      int gRow = tileXstart + tx;
      int gCol = tileYstart + ty;

      Tmapping root =
          s_tiles.get(gRow, gCol, borderTileYstart, borderTileXstart);

      // Now iterate over the window, again via global indices
      for (int gwy = gRow - windowDist.y; gwy <= gRow + windowDist.y; gwy++) {
        for (int gwx = gCol - windowDist.x; gwx <= gCol + windowDist.x; gwx++) {
          // If the element in the window is inside the 'main' tile then ignore
          // it
          if (gwy >= tileYstart && gwy < tileYstart + tileDims.y &&
              gwx >= tileXstart && gwx < tileXstart + tileDims.x) {
            continue;
          }

          // Otherwise it is inside the border, test it
          Tmapping windowRoot =
              s_tiles.get(gwy, gwx, borderTileYstart, borderTileXstart);

          // Take min root
          root = root < windowRoot ? root : windowRoot;

          // TODO: this doesn't really work, i need to hold all neighbours to
          // the current element and append them somewhere, while maintaining
          // only unique windowRoots.. pause while i figure this out
        }
      }
    }
  }
}

template <typename Tmapping>
__global__ void naive_global_unionfind_kernel(DeviceImage<Tmapping> mapping,
                                              const int2 tileDims,
                                              const int2 windowDist) {
  // Still 'work in the tile'
  const int tileXstart = blockIdx.x * tileDims.x;
  const int tileYstart = blockIdx.y * tileDims.y;
  const int2 blockTileDims{tileXstart + tileDims.x < (int)mapping.width
                               ? tileDims.x
                               : (int)mapping.width - tileXstart,
                           tileYstart + tileDims.y < (int)mapping.height
                               ? tileDims.y
                               : (int)mapping.height - tileYstart};

  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      // Don't do anything if it is not in range of the border
      if (tx < windowDist.x || tx >= blockTileDims.x - windowDist.x ||
          ty < windowDist.y || ty >= blockTileDims.y - windowDist.y) {
        continue;
      }

      // Read the value, which is the tile root
      int gy = tileYstart + ty;
      int gx = tileXstart + tx;
      Tmapping *rootPtr = find(
          mapping, (Tmapping)(gy * mapping.width + gx)); // this will not change
      // Ignore inactive sites
      if (rootPtr == nullptr) {
        continue;
      }
      // Unite with neighbours in the border
      for (int gwy = ty - windowDist.y; gwy <= ty + windowDist.y; gwy++) {
        for (int gwx = tx - windowDist.x; gwx <= tx + windowDist.x; gwx++) {
          // Ignore if outside image
          if (gwy < 0 || gwy >= (int)mapping.height || gwx < 0 ||
              gwx >= (int)mapping.width) {
            continue;
          }
          // Ignore inside the tile
          if (gwy >= tileYstart && gwy < tileYstart + blockTileDims.y &&
              gwx >= tileXstart && gwx < tileXstart + blockTileDims.x) {
            continue;
          }
          // Ignore if inactive
          Tmapping *neighbourRootPtr =
              find(mapping, (Tmapping)(gwy * mapping.width + gwx));
          if (neighbourRootPtr == nullptr) {
            continue;
          }

          // Otherwise change the root addresses
          if (*rootPtr < *neighbourRootPtr) {
            // Change the neighbour's root to this element's root
            printf("Changing root %d -> %d\n", *neighbourRootPtr, *rootPtr);
            atomicMin(neighbourRootPtr, *rootPtr);
          } else if (*rootPtr > *neighbourRootPtr) {
            // Change this element's root to the neighbour's root
            printf("Changing root %d -> %d\n", *rootPtr, *neighbourRootPtr);
            atomicMin(rootPtr, *neighbourRootPtr);
          }
        }
      }
    }
  }
}

} // namespace wccl

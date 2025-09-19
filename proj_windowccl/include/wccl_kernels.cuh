#pragma once

#include <cstdint>
#include <stdexcept>
#include <thrust/device_vector.h>

#include "accessors.cuh"
#include "atomic_extensions.cuh"
#include "containers/bitset.cuh"
#include "sharedmem.cuh"

namespace wccl {

#ifndef NDEBUG
#define dprintf(...) printf(__VA_ARGS__)
#define d1printf(...)                                                          \
  if (threadIdx.x == 0 && threadIdx.y == 0)                                    \
  printf(__VA_ARGS__)
#else
#define dprintf(...)
#define d1printf(...)
#endif

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

  __host__ __device__ T *getPointer(const unsigned int row,
                                    const unsigned int col) {
    return &data[flattenedIndex(row, col)];
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
 * @brief Finder for flattened 1D index images. May be highly warp divergent.
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

template <typename Tmapping>
__device__ void pathcompress(DeviceImage<Tmapping> &img, Tmapping idx) {
  Tmapping *rootPtr = find(img, idx);
  if (img.data[idx] != *rootPtr)
    img.data[idx] = *rootPtr;
}

// /**
//  * @brief Finder for 2D col/row pair index images. May be highly warp
//  divergent.
//  */
// template <typename Tcolrow>
// __device__ Tcolrow *find(const DeviceIndexImage<Tcolrow> &tile, Tcolrow idx)
// {
//   if (tile.isInvalidAt(idx.y, idx.x))
//     return nullptr;
//
//   while (!tile.isSelfRoot(idx.y, idx.x)) {
//     idx = tile.get(idx.y, idx.x);
//   }
//   return tile.getPointer(idx.y, idx.x);
// }

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

/**
 * @brief Loads and initializes the tile in shared memory with flattened indices
 * based on the input's active sites. Additionally updates the book-keeper with
 * active site indices.
 *
 * @tparam Tmapping Type of the mapping indices
 * @param input Input image, from global memory
 * @param s_tile Shared memory tile struct
 * @param blockTileDims Dimensions of the tile for this block (accounting for
 * edge)
 * @param tileXstart X (col) offset of the tile
 * @param tileYstart Y (row) offset of the tile
 * @param s_activeIdx Shared memory book-keeper for active indices (appending
 * list). Ignored if nullptr (default).
 * @param s_numActiveSites Shared memory book-keeper for number of active sites
 * (length of list). Ignored if nullptr (default).
 */
template <typename Tmapping>
__device__ void initializeShmemTile(const DeviceImage<uint8_t> &input,
                                    DeviceImage<Tmapping> &s_tile,
                                    const int2 blockTileDims,
                                    const int tileXstart, const int tileYstart,
                                    Tmapping *s_activeIdx = nullptr,
                                    unsigned int *s_numActiveSites = nullptr) {
  // Load the data into the tile
  bool validX, validY;
  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    // Define global index for reads (row)
    int y = tileYstart + ty;
    // Ignore out of bounds
    validY = y < (int)input.height;

    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      // Define global index for reads (col)
      int x = tileXstart + tx;
      // Ignore out of bounds
      validX = x < (int)input.width;

      // Global read; if inactive (or out of bounds), set to -1, otherwise
      // set to flattened local index (for now)
      const uint8_t site = (validX && validY) ? input.get(y, x) : 0;
      if (site > 0) {
        Tmapping flatIdx = s_tile.flattenedIndex(ty, tx);
        s_tile.set(ty, tx, flatIdx);
        if (s_activeIdx != nullptr && s_numActiveSites != nullptr)
          // Add to our bookkeeper of active sites
          s_activeIdx[atomicAggInc(s_numActiveSites)] = flatIdx;
      } else
        s_tile.set(ty, tx, -1);

      dprintf("Init tile(%d,%d)<-input.get(%d,%d): %d\n", ty, tx, y, x,
              s_tile.get(ty, tx));
    }
  }
}

template <typename Tmapping>
__device__ void unite(DeviceImage<Tmapping> &s_tile, Tmapping *rootPtr,
                      Tmapping windowIndex, Tmapping &counter) {
  // Read the window element's root address
  Tmapping *wrootPtr = find(s_tile, windowIndex);
  dprintf("Blk (%d,%d): %d, %d -> window root %d\n", blockIdx.x, blockIdx.y, wx,
          wy, *rootPtr);
  // // Mini path compression (not much difference?)
  // if (s_tile.get(wy, wx) > *wrootPtr)
  //   atomicMin(s_tile.getPointer(wy, wx), *wrootPtr);

  if (*wrootPtr < *rootPtr) {
    // Change current root pointer
    atomicMin(rootPtr, *wrootPtr);
    counter++;
    // we should shift our root pointer to the new root pointer
    rootPtr = wrootPtr;
  } else if (*wrootPtr > *rootPtr) {
    // Change window root pointer
    atomicMin(wrootPtr, *rootPtr);
    counter++;
  }
}

// ========================================================================
// ========================================================================
// ====================== GLOBAL KERNELS ==================================
// ========================================================================
// ========================================================================

template <typename Tmapping>
__global__ void local_connect_naive_unionfind_kernel(
    const DeviceImage<uint8_t> input, DeviceImage<Tmapping> mapping,
    const int2 tileDims, const int2 windowDist) {
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
  SharedMemory<unsigned int> smem;
  // First value is just a counter
  unsigned int *s_numActiveSites = smem.getPointer();
  // Single workspace for the tile
  DeviceImage<Tmapping> s_tile((Tmapping *)&s_numActiveSites[1],
                               blockTileDims.y, blockTileDims.x);
  // and a workspace to bookkeep active indices
  Tmapping *s_activeIdx = &s_tile.data[s_tile.size()];

  // Initialize counters
  if (threadIdx.y == 0) {
    if (threadIdx.x == 0)
      *s_numActiveSites = 0;
  }
  __syncthreads();

  // Load the data into the tile
  initializeShmemTile(input, s_tile, blockTileDims, tileXstart, tileYstart,
                      s_activeIdx, s_numActiveSites);
  __syncthreads();

  // Now attempt to unite sets for real
  unsigned int numActiveSites = *s_numActiveSites;
  int counter = 0; // local thread-register counter
  bool continueIterations = true;
  while (continueIterations) {
    counter = 0; // reset local thread-register counter
    // Iterate over active sites
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < (int)numActiveSites; i += blockDim.y * blockDim.x) {
      int ty = s_activeIdx[i] / blockTileDims.x;
      int tx = s_activeIdx[i] % blockTileDims.x;

      // // Iterate over entire tile
      // for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
      //   for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      // If not valid, ignore
      // if (s_tile.get(ty, tx) < 0)
      //   continue;

      // Read the current element's root address
      Tmapping *rootPtr = find(s_tile, (Tmapping)s_tile.flattenedIndex(ty, tx));
      // // Mini path compression (not much difference?)
      // if (s_tile.get(ty, tx) > *rootPtr)
      //   atomicMin(s_tile.getPointer(ty, tx), *rootPtr);

      dprintf("Blk (%d,%d): %d, %d -> initial root %d\n", blockIdx.x,
              blockIdx.y, tx, ty, *rootPtr);

      // Iterate over window
      for (int wy = ty - windowDist.y; wy <= ty + windowDist.y; wy++) {
        if (wy < 0 || wy >= s_tile.height)
          continue;

        for (int wx = tx - windowDist.x; wx <= tx + windowDist.x; wx++) {
          if (wx < 0 || wx >= s_tile.width)
            continue;

          // Ignore if the window element is invalid
          if (s_tile.get(wy, wx) < 0)
            continue;

          unite(s_tile, rootPtr, (Tmapping)s_tile.flattenedIndex(wy, wx),
                counter);
        }
      }
      //   } // end loop over tile (x)
      // } // end loop over tile (y)
    } // end loop over active sites

    // Check if any thread in the block has made any changes
    int blocksynced = __syncthreads_or(counter);
    continueIterations = blocksynced != 0;
  } // end while

  // Path compression, then write to global
  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      Tmapping element = s_tile.get(ty, tx);
      // Path compress if valid
      if (element >= 0) {
        // pathcompress(s_tile, (Tmapping)s_tile.flattenedIndex(ty, tx));
        // element = s_tile.get(ty, tx);
        // Just read the root directly?
        Tmapping *rootPtr =
            find(s_tile, (Tmapping)s_tile.flattenedIndex(ty, tx));
        element = *rootPtr;
        // Split 1D index into col and row
        int gcol = element % s_tile.width + tileXstart;
        int grow = element / s_tile.width + tileYstart;
        element = (Tmapping)mapping.flattenedIndex(grow, gcol);
      }
      // Write the element
      mapping.set(ty + tileYstart, tx + tileXstart, element);
    }
  }
}

/**
 * @brief Helper wrapper function to handle grid dimensions and shared memory
 * for local_connect_naive_unionfind_kernel.
 *
 * @return Grid dimensions, which contains the number of tiles calculated.
 */
template <typename Tmapping>
dim3 local_connect_naive_unionfind(const DeviceImage<uint8_t> input,
                                   DeviceImage<Tmapping> mapping,
                                   const int2 tileDims, const int2 windowDist,
                                   const dim3 tpb) {
  dim3 bpg(input.width / tileDims.x + (input.width % tileDims.x > 0 ? 1 : 0),
           input.height / tileDims.y + (input.height % tileDims.y > 0 ? 1 : 0));
  size_t shmem = tileDims.x * tileDims.y *
                     (sizeof(Tmapping) * 2) + // tile + active index bookkeeping
                 1 * sizeof(unsigned int);    // counter for active indices

  wccl::local_connect_naive_unionfind_kernel<Tmapping>
      <<<bpg, tpb, shmem>>>(input, mapping, tileDims, windowDist);

  return bpg;
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

#ifndef NDEBUG
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    dprintf("numUniteSteps: %d\n", numUniteSteps);
    //   for (int i = 0; i < blockTileDims.y; ++i) {
    //     for (int j = 0; j < blockTileDims.x; ++j) {
    //       printf("(%d, %d): (%d, %d)\n", i, j,
    //              s_tiles[numUniteSteps % 2].get(i, j).y,
    //              s_tiles[numUniteSteps % 2].get(i, j).x);
    //     }
    //   }
  }
#endif

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

// template <typename Tmapping>
// __global__ void inter_tile_neighbours_kernel(DeviceImage<Tmapping> mapping,
//                                              const int2 tileDims,
//                                              const int2 windowDist) {
//   static_assert(std::is_signed_v<Tmapping>, "mapping type T must be signed");
//
//   // Define the tile for this block
//   const int tileXstart = blockIdx.x * tileDims.x;
//   const int tileYstart = blockIdx.y * tileDims.y;
//   // We then define an enlarged tile to include the border (this is what we
//   will
//   // load)
//   const int borderTileXstart =
//       tileXstart - windowDist.x < 0 ? 0 : tileXstart - windowDist.x;
//   const int borderTileYstart =
//       tileYstart - windowDist.y < 0 ? 0 : tileYstart - windowDist.y;
//   // Tile may not be fully occupied at the ends
//   const int2 blockTileDims{tileXstart + tileDims.x < (int)mapping.width
//                                ? tileDims.x
//                                : (int)mapping.width - tileXstart,
//                            tileYstart + tileDims.y < (int)mapping.height
//                                ? tileDims.y
//                                : (int)mapping.height - tileYstart};
//
//   const int2 borderBlockTileDims{
//       borderTileXstart + tileDims.x + 2 * windowDist.x < (int)mapping.width
//           ? tileDims.x + 2 * windowDist.x
//           : (int)mapping.width - borderTileXstart,
//       borderTileYstart + tileDims.y + 2 * windowDist.y < (int)mapping.height
//           ? tileDims.y + 2 * windowDist.y
//           : (int)mapping.height - borderTileYstart,
//   };
//
//   // Read tile with border into shared memory
//   SharedMemory<Tmapping> smem;
//   DeviceIndexImage<Tmapping> s_tiles(smem.getPointer(),
//   borderBlockTileDims.y,
//                                      borderBlockTileDims.x);
//
//   blockRoiLoad(mapping.data, mapping.width, mapping.height, borderTileXstart,
//                borderBlockTileDims.x, borderTileYstart,
//                borderBlockTileDims.y, s_tiles.data);
//   __syncthreads();
//
//   // Now for every element, check for neighbours
//   for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
//     for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
//       // We read elements based on their global indices instead of the local
//       // ones, to track the internal tile WITHOUT the borders
//       int gRow = tileXstart + tx;
//       int gCol = tileYstart + ty;
//
//       Tmapping root =
//           s_tiles.get(gRow, gCol, borderTileYstart, borderTileXstart);
//
//       // Now iterate over the window, again via global indices
//       for (int gwy = gRow - windowDist.y; gwy <= gRow + windowDist.y; gwy++)
//       {
//         for (int gwx = gCol - windowDist.x; gwx <= gCol + windowDist.x;
//         gwx++) {
//           // If the element in the window is inside the 'main' tile then
//           ignore
//           // it
//           if (gwy >= tileYstart && gwy < tileYstart + tileDims.y &&
//               gwx >= tileXstart && gwx < tileXstart + tileDims.x) {
//             continue;
//           }
//
//           // Otherwise it is inside the border, test it
//           Tmapping windowRoot =
//               s_tiles.get(gwy, gwx, borderTileYstart, borderTileXstart);
//
//           // Take min root
//           root = root < windowRoot ? root : windowRoot;
//
//           // TODO: this doesn't really work, i need to hold all neighbours to
//           // the current element and append them somewhere, while maintaining
//           // only unique windowRoots.. pause while i figure this out
//         }
//       }
//     }
//   }
// }

template <typename Tmapping>
__global__ void naive_global_unionfind_kernel(DeviceImage<Tmapping> mapping,
                                              const int2 tileDims,
                                              const int2 windowDist,
                                              unsigned int *updateCounter) {
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
    // Must be in range of border
    /*
    E.g., only O is considered, windowDist = {2, 2}
    OOOOOOO
    OOOOOOO
    OOXXXOO
    OOOOOOO
    OOOOOOO
    */
    bool validY = false;
    if (ty < windowDist.y || ty >= blockTileDims.y - windowDist.y)
      validY = true;

    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      bool validX = false;
      if (tx < windowDist.x || tx >= blockTileDims.x - windowDist.x)
        validX = true;

      // Don't do anything if it is not in range of the border
      if (!(validX || validY))
        continue;

      // Translate tile index to global index
      int gy = tileYstart + ty;
      int gx = tileXstart + tx;
      dprintf("Blk (%d,%d): %d, %d == %d, %d\n", blockIdx.x, blockIdx.y, tx, ty,
              gx, gy);

      // Ignore out of range of global image
      if (gy < 0 || gy >= (int)mapping.height || gx < 0 ||
          gx >= (int)mapping.width) {
        continue;
      }

      // Read the value, which is the tile root
      Tmapping *rootPtr = find(
          mapping, (Tmapping)(gy * mapping.width + gx)); // this will not change
      // Ignore inactive sites
      if (rootPtr == nullptr) {
        continue;
      }
      // Unite with neighbours in the border
      for (int gwy = gy - windowDist.y; gwy <= gy + windowDist.y; gwy++) {
        // Ignore if outside image
        if (gwy < 0 || gwy >= (int)mapping.height) {
          continue;
        }
        for (int gwx = gx - windowDist.x; gwx <= gx + windowDist.x; gwx++) {
          // Ignore if outside image
          if (gwx < 0 || gwx >= (int)mapping.width) {
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
            dprintf("Blk (%d,%d), pos(%d, %d), wpos(%d,%d): Tile wants to "
                    "change neighbour "
                    "root %p(%d) -> "
                    "%p(%d)\n",
                    blockIdx.x, blockIdx.y, gx, gy, gwx, gwy, neighbourRootPtr,
                    *neighbourRootPtr, rootPtr, *rootPtr);
            atomicMin(neighbourRootPtr, *rootPtr);
            atomicAdd(updateCounter, 1);
          } else if (*rootPtr > *neighbourRootPtr) {
            // Change this element's root to the neighbour's root
            dprintf("Blk (%d,%d), pos(%d, %d), wpos(%d,%d): Neighbour wants to "
                    "change tile "
                    "root %p(%d) -> "
                    "%p(%d)\n",
                    blockIdx.x, blockIdx.y, gx, gy, gwx, gwy, rootPtr, *rootPtr,
                    neighbourRootPtr, *neighbourRootPtr);
            atomicMin(rootPtr, *neighbourRootPtr);
            atomicAdd(updateCounter, 1);
            // modify our rootPtr
            rootPtr = neighbourRootPtr;
          }
        }
      }
    }
  }
}

template <typename Tbitset> struct NeighbourChainer {
  // single row in neighbour matrix alpha (the seed row being worked on)
  containers::Bitset<Tbitset, int> seedRow;
  // single row in neighbour matrix alpha (the row of the neighbour)
  containers::Bitset<Tbitset, int> neighbourRow;
  containers::Bitset<Tbitset, int> beta;  // availability vector
  containers::Bitset<Tbitset, int> gamma; // neighbour of interest vector
  int earliestValidBetaIndex = 0;

  __host__ __device__ NeighbourChainer() {};
  __host__ __device__ NeighbourChainer(Tbitset *_seedRow,
                                       Tbitset *_neighbourRow, Tbitset *_beta,
                                       Tbitset *_gamma, const int numPixels)
      : seedRow(_seedRow, numPixels), neighbourRow(_neighbourRow, numPixels),
        beta(_beta, numPixels), gamma(_gamma, numPixels) {}
  /**
   * @brief Constructs a NeighbourChainer object with some remaining buffer,
   usually the shared memory workspace.
   * This assumes that the remaining buffer has sufficient allocation for all 4
   (identically sized) Bitsets, which should be determined from the
   numElementsRequiredFor static method.
   *
   * @param _data Pointer to remaining buffer (e.g. in shared memory)
   * @param numPixels Number of pixels (bits) in the image (bitset)
   */
  __host__ __device__ NeighbourChainer(Tbitset *_data, const int numPixels)
      : seedRow(_data, numPixels),
        neighbourRow(
            &_data[containers::Bitset<Tbitset, int>::numElementsRequiredFor(
                numPixels)],
            numPixels),
        beta(&_data[containers::Bitset<Tbitset, int>::numElementsRequiredFor(
                        numPixels) *
                    2],
             numPixels),
        gamma(&_data[containers::Bitset<Tbitset, int>::numElementsRequiredFor(
                         numPixels) *
                     3],
              numPixels) {}

  __host__ __device__ bool validFor(const DeviceImage<uint8_t> &img) const {
    return img.size() == seedRow.numBits;
  }

  __device__ void setEarliestValidBetaIndex() {
    // TODO: not sure if really need to do parallel reduction just for this?

    // We start at the previous earliestValidBetaIndex + 1
    while (earliestValidBetaIndex < beta.numBits) {
      if (beta.getBitAt(earliestValidBetaIndex))
        break;

      earliestValidBetaIndex++;
    }
  }

  __device__ bool isComplete() const {
    return earliestValidBetaIndex == beta.numBits;
  }

  template <typename Tmapping>
  __device__ void blockInitBeta(const DeviceImage<Tmapping> &img) {
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < beta.numDataElements; i += blockDim.y * blockDim.x) {
      // Each thread works on one element word of the bitset (so its 8/16/32/64
      // bits depending on Tbitset)

      // Set individual bits if active (no need to initialize, we write
      // everything)
      // TODO: should optimise this to prevent shared mem bank conflicts
      for (int j = 0; j < beta.numBitsPerElement(); j++) {
        int pixelFlatIdx = i * beta.numBitsPerElement() + j;
        if (pixelFlatIdx < img.size())
          beta.setBitAt(pixelFlatIdx, img.data[pixelFlatIdx] >= 0);
        else
          beta.setBitAt(pixelFlatIdx, 0);
      }
    }
  }

  /**
   * @brief Queries the availability vector for the specified row. If the row is
   * not available, it should not be computed via blockComputeAlphaRow().
   *
   * @param rowIndex Target row index
   * @return True if the row is available
   */
  __device__ bool isAvailable(const int rowIndex) {
    return beta.getBitAt(rowIndex);
  }

  /**
   * @brief Computes a single row in the alpha matrix. Used to fill either the
   * alphaRow or workspaceRow. You may need to syncthreads() after this.
   *
   * @param img Initial input image to read from (identical to union find method
   * load)
   * @param windowDist Window distance, to determine neighbours
   * @param targetIndex Target index inside the alpha matrix. For
   */
  template <typename Tmapping>
  __device__ void
  blockComputeAlphaRow(const DeviceImage<Tmapping> &img, const int2 windowDist,
                       const int targetIndex,
                       containers::Bitset<Tbitset, int> &alphaRow) {
    // Position inside the 2D image
    int targetRow = targetIndex / img.width;
    int targetCol = targetIndex % img.width;

    // Each thread computes an element in the row
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < alphaRow.numDataElements; i += blockDim.y * blockDim.x) {
      for (int j = 0; j < alphaRow.numBitsPerElement(); j++) {
        int pixelFlatIdx = i * alphaRow.numBitsPerElement() + j;
        // Ignore out of bounds, or if inactive
        if (pixelFlatIdx > img.size() || img.data[pixelFlatIdx] < 0) {
          alphaRow.setBitAt(pixelFlatIdx, 0);
          continue;
        }
        int pixelRow = pixelFlatIdx / img.width;
        int pixelCol = pixelFlatIdx % img.width;

        // Set 1 if this is within the window
        if (abs(pixelRow - targetRow) <= windowDist.y &&
            abs(pixelCol - targetCol) <= windowDist.x) {

          alphaRow.setBitAt(pixelFlatIdx, 1);
          dprintf("Examining if %d is neighbour to row %d: yes\n", pixelFlatIdx,
                  targetIndex);
        } else {

          alphaRow.setBitAt(pixelFlatIdx, 0);
          dprintf("Examining if %d is neighbour to row %d: no\n", pixelFlatIdx,
                  targetIndex);
        }
      }
    }
  }

  __device__ void consumeAlphaRow(const int rowIndex) {
    if (threadIdx.x == 0 && threadIdx.y == 0)
      beta.setBitAt(rowIndex, 0);
  }

  /**
   * @brief Computes the gamma vector (neighbours of interest). Explicitly
   * performs the syncthreads() internally, since this is required to determine
   * if it is non-empty.
   *
   * @return True if gamma is non-empty.
   */
  __device__ bool blockComputeNeighboursOfInterest() {
    int threadFoundNeighbours = 0;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < gamma.numDataElements; i += blockDim.y * blockDim.x) {
      // Simply bitwise AND everything
      gamma.elementAt(i) = seedRow.elementAt(i) & beta.elementAt(i);
      if (gamma.elementAt(i) != 0)
        threadFoundNeighbours = 1;
    }
    int hasNeighbours = __syncthreads_or(threadFoundNeighbours);

    return hasNeighbours != 0;
  }

  __device__ void blockMergeRows() {
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < gamma.numDataElements; i += blockDim.y * blockDim.x) {
      // Simply bitwise OR everything
      seedRow.elementAt(i) = seedRow.elementAt(i) | neighbourRow.elementAt(i);
    }
  }

  template <typename Tmapping>
  __device__ void blockChainNeighbours(const int2 windowDist,
                                       DeviceImage<Tmapping> &img) {
    static_assert(std::is_signed_v<Tmapping>, "Tmapping must be signed");
    // Compute the current seed row
    blockComputeAlphaRow(img, windowDist, earliestValidBetaIndex, seedRow);
    consumeAlphaRow(earliestValidBetaIndex);
    d1printf("seed row at %d\n", earliestValidBetaIndex);
    // since every thread works on the same element in seedRow/gamma, there is
    // no need to syncthreads yet

    // // debug sync
    // __syncthreads();
    // for (int i = 0; i < gamma.numBits; ++i) {
    //   d1printf("seedRow[%d]: %d\n", i, seedRow.getBitAt(i));
    // }

    // Compute gamma
    bool gammaNonEmpty = blockComputeNeighboursOfInterest();
    // we sync here because the whole block needs to know the neighbours of
    // interest
    for (int i = 0; i < gamma.numBits; ++i) {
      d1printf("gamma[%d]: %d\n", i, gamma.getBitAt(i));
    }

    while (gammaNonEmpty) {
      // Iterate over gamma for this seed row
      for (int n = 0; n < gamma.numBits; ++n) {
        // If not of interest, ignore
        if (!gamma.getBitAt(n))
          continue;
        d1printf("gamma neighbour for row %d -> %d\n", earliestValidBetaIndex,
                 n);

        // Otherwise we compute the row for this neighbour
        blockComputeAlphaRow(img, windowDist, n, neighbourRow);
        consumeAlphaRow(n);

        // And then we bitwise OR it into the seed row
        blockMergeRows();
        // no need to sync, each row is independent, and again, threads work on
        // same index on all rows
      }
      __syncthreads();

      // Recompute gamma again
      gammaNonEmpty = blockComputeNeighboursOfInterest();
    }
    // Once gamma is complete, our seed row is fully merged and ready to be
    // output. NOTE: gamma calculation already synced for us
    for (int i = threadIdx.y; i < img.height; i += blockDim.y) {
      for (int j = threadIdx.x; j < img.width; j += blockDim.x) {
        int idx = i * img.width + j;
        if (seedRow.getBitAt(idx))
          img.set(i, j,
                  earliestValidBetaIndex); // set all to the flattened index
      }
    }

    // Update the earliest index to find next one
    setEarliestValidBetaIndex();
    d1printf("after blockChainNeighbours, new earliest index: %d\n",
             earliestValidBetaIndex);
  }
}; // end struct NeighbourChainer

template <typename Tbitset, typename Tmapping>
__global__ void localChainNeighboursKernel(const DeviceImage<Tbitset> input,
                                           const int2 windowDist,
                                           const int2 tileDims,
                                           DeviceImage<Tmapping> output) {
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
  SharedMemory<Tmapping> smem;
  // Workspace for the tile
  DeviceImage<Tmapping> s_tile(smem.getPointer(), blockTileDims.y,
                               blockTileDims.x);
  // Workspaces for the neighbour chainer struct
  NeighbourChainer<Tbitset> s_chainer((Tbitset *)&s_tile.data[s_tile.size()],
                                      blockTileDims.x * blockTileDims.y);

  // Read the tile and populate (same as unionfind)
  initializeShmemTile(input, s_tile, blockTileDims, tileXstart, tileYstart);
  __syncthreads();

  // And initialize our neighbour chainer
  s_chainer.template blockInitBeta<Tmapping>(s_tile);
  __syncthreads();

  // Fill a register-local first beta
  s_chainer.setEarliestValidBetaIndex();
  d1printf("initial earliest beta index: %d\n",
           s_chainer.earliestValidBetaIndex);

  // Now run the consumption
  while (!s_chainer.isComplete()) {
    d1printf("blockChain iteration\n");
    s_chainer.blockChainNeighbours(windowDist, s_tile);
  }
  __syncthreads();

  // Then once done we output back to global
  for (int ty = threadIdx.y; ty < blockTileDims.y; ty += blockDim.y) {
    for (int tx = threadIdx.x; tx < blockTileDims.x; tx += blockDim.x) {
      Tmapping element = s_tile.get(ty, tx);
      if (element >= 0) {
        // Split 1D index into col and row
        int gcol = element % s_tile.width + tileXstart;
        int grow = element / s_tile.width + tileYstart;
        element = (Tmapping)output.flattenedIndex(grow, gcol);
      }
      // Write the element
      output.set(ty + tileYstart, tx + tileXstart, element);
    }
  }
}

} // namespace wccl

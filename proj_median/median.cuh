#pragma once

#include <cstdint>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cuda/std/limits>

#include "atomic_extensions.cuh"
#include "sharedmem.cuh"

#include "containers/image.cuh"

/**
 * @brief Performs a row-wise median of a 2D input array, with one block
 * tackling each row. Each row may have a different occupied length, as
 * specified by a separate array.
 *
 * Fundamentally, this performs a cub::BlockRadixSort, where it sets the unused
 * elements to be numeric_limits<T>::max(). This allows us to extract the
 * 'correct' median by referencing the valid length.
 *
 * NOTE: median is defined to be a simple length/2, so even-length rows may not
 * be as expected, but it is more efficient to process this way.
 *
 * @tparam T Type of input data
 * @param inputRows The input data
 * @param numRows The number of input rows
 * @param numColumns The (max) number of input columns, not all may be used
 * @param inputLengths The number of actually used elements in each row
 * @param medians Output median values for each row
 */
template <typename T, int NUM_THREADS_PER_BLK, int ELEM_PER_THREAD,
          bool allValidInFront = true>
__global__ void blockwise_median_kernel(
    const T* inputRows,      // the input data, numRows x numColumns
    const int numRows,       // number of rows
    const int numColumns,    // number of columns
    const int* inputLengths, // the number of actually used elements in each row
    T* medians               // numRows
) {
  // Define the sort length
  constexpr int SORT_LENGTH = NUM_THREADS_PER_BLK * ELEM_PER_THREAD;

  // Each block processes one row
  int row = blockIdx.x;
  // Exit if nothing to do
  if (row >= numRows)
    return;

  // Read the valid input length for this row
  const int validLength = inputLengths[row];

  // Check whether the sort length is valid (must be more than the input length
  // of this row) Otherwise write the max value for this type
  if (SORT_LENGTH < validLength) {
    if (threadIdx.x == 0) {
      medians[row] = cuda::std::numeric_limits<T>::max();
    }
    return;
  }

  // Special case: if the length is 0, we return without writing anything
  // Otherwise, due to how we fill the remainder elements, it will write
  // numeric_limits<T>::max() instead.
  if (validLength == 0)
    return;

  // Extract the row that will be processed
  size_t rowOffset = row * numColumns; // may be very large, so use size_t here
  const T* inputRow = &inputRows[rowOffset];

  // Standard cub boilerplate, adapted from example_block_radix_sort.cu
  using BlockRadixSort =
      cub::BlockRadixSort<T, NUM_THREADS_PER_BLK, ELEM_PER_THREAD>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  // Per-thread tile items
  T items[ELEM_PER_THREAD];
  // Fill our per-thread tile items, globally coalesced
  // 1st loop fills index 0, 2nd loop fills index 1, etc
  // There should be a total of ELEM_PER_THREAD loops
  for (int t = threadIdx.x; t < NUM_THREADS_PER_BLK * ELEM_PER_THREAD;
       t += NUM_THREADS_PER_BLK) {
    // Compile-time switch based on the type of data supplied
    // Here we assume that the first validLength elements of each row are used
    // E.g. O O O O O X X ...
    if constexpr (allValidInFront) {
      // we write the valid elements from the front
      if (t < validLength)
        items[t / NUM_THREADS_PER_BLK] = inputRow[t];
      // Fill the remainder with the max type value
      else
        items[t / NUM_THREADS_PER_BLK] = cuda::std::numeric_limits<T>::max();
    }
    // Otherwise, we cannot assume the first validLength elements are to be used
    // and we instead assume that the invalid elements have already been marked
    // as numeric_limits<T>::max(), so we read all the elements available
    // E.g. O O X O O X O X O ...
    else {
      if (t < numColumns)
        items[t / NUM_THREADS_PER_BLK] = inputRow[t];
      // Fill the remainder with the max type value
      else
        items[t / NUM_THREADS_PER_BLK] = cuda::std::numeric_limits<T>::max();
    }
  }

  // Now sort
  BlockRadixSort(temp_storage).Sort(items);

  // The median should be in the element corresponding to half valid length
  // We choose to ignore the even-length case and just use half valid length
  const int medianIdx = validLength / 2;
  const int threadIdxContainingMedian = medianIdx / ELEM_PER_THREAD;
  if (threadIdx.x == threadIdxContainingMedian)
    medians[row] = items[medianIdx % ELEM_PER_THREAD];
}

template <typename T, int MAX_LENGTH> struct BlockQuickSelect {
  T sdata[MAX_LENGTH * 2];
  int scounters[2];

  __device__ void fill(int usedLength, const T* d_row) {
    for (int t = threadIdx.x; t < usedLength; t += blockDim.x) {
      if (t < MAX_LENGTH)
        this->sdata[t] = d_row[t];
    }
    __syncthreads();
  }

  __device__ T select(int length, int n) {
    // Entire data for the block starts in first half of sdata

    T* sdata1 = sdata;
    T* sdata2 = &sdata1[MAX_LENGTH];

    T pivot;
    int pivotIdx;
    int offset = 0;
    // Iterations should never exceed this anyway, just as a soft cap
    for (int i = 0; i < MAX_LENGTH; ++i) {
      // Get pivot, just use first element
      pivotIdx = i % length;
      pivot = sdata1[pivotIdx + offset];

      // Reset counters
      if (threadIdx.x == 0)
        scounters[0] = 0;
      if (threadIdx.x == 1)
        scounters[1] = 0;
      __syncthreads();

      // Loop over the side we want to recurse into
      for (int t = threadIdx.x; t < length; t += blockDim.x) {
        // Skip pivot
        if (t == pivotIdx)
          continue;

        // Read current element to compare
        // There is an additional offset because we may be reading from
        // the back loaded section of a previous iteration e.g.
        // LLLLXXXRRRRRRRR
        //        └start of the array for the right side
        T element = sdata1[t + offset];

        // Check <
        if (element < pivot) {
          // Front load to 2nd buffer
          int oIdx = atomicAggInc(&scounters[0]);
          sdata2[oIdx] = element;
        }
        // Check >
        else {
          // Back load to 2nd buffer
          int oIdx = MAX_LENGTH - atomicAggInc(&scounters[1]) - 1;
          sdata2[oIdx] = element;
        }
      }
      __syncthreads();

      // Which side do we recurse to?
      if (n == scounters[0]) {
        // e.g. looking for index 5, and there are 5
        // elements on left, then pivot is the selection
        break;
      } else if (n < scounters[0]) {
        // e.g. looking for index 5, and there are 6 elements on left i.e. 0-5
        // then element we want is on the left
        length = scounters[0];
        // no need to change 'n' since the 'n'th element in the whole is still
        // the 'n'th in the left
        // offset must now be 0
        offset = 0;
      } else {
        // e.g. looking for index 5, and there are 4 elements on left i.e. 0-3
        // then element we want is on the right
        length = scounters[1];
        // change the 'n' in the recursed shorter array on the right
        // in above example, 'n' would now be 0 [(0-3), pivot, target, ...)]
        n = n - scounters[0] - 1;
        // make sure our offset is now to the start of the right side
        offset = MAX_LENGTH - scounters[1];
      }
      __syncthreads();

      swapBuffers(&sdata1, &sdata2);
    } // end while loop

    // if we reach this without getting a value, use a helpful error value
    return pivot;
  } // end select() function

  __device__ void swapBuffers(T** sdata1, T** sdata2) {
    T* spare = *sdata2;
    *sdata2 = *sdata1;
    *sdata1 = spare;
  }
};

template <typename T, int MAX_LENGTH>
__global__ void blockwise_quickselect_kernel(const T* d_input, int numRows,
                                             int numCols, int* lengths,
                                             int* d_n, T* d_output) {
  // Get the row for this block
  int row = blockIdx.x;
  if (row >= numRows)
    return;
  const T* d_row = &d_input[row * numCols];
  // And the actual used length and selected element
  int usedLength = lengths[row];
  int n = d_n[row];

  // Call quickselect (now templated for max length with shared memory
  // allocation directly)
  __shared__ BlockQuickSelect<T, MAX_LENGTH> selector;
  selector.fill(usedLength, d_row);
  T nthElement = selector.select(usedLength, n);
  if (threadIdx.x == 0)
    d_output[row] = nthElement;
}

// =============================================
// =============================================
// =============================================
// ====== ROW-WISE 1D MEDIAN FILTER ============
// =============================================
// =============================================

/**
 * @brief Warp-distributed rank state for a sliding median window.
 *
 * @details One instance exists per lane, and the 32 lane-local instances
 * cooperate over one shared rank array. Each rank remains attached to a fixed
 * ring slot. Window values remain in shared memory; `head` and the current
 * window pointer recover the shared-memory value belonging to any ring slot.
 * All methods must be called convergently by a complete warp.
 *
 * `Tplane` must be large enough that `Tplane * 32 >= windowLen`. The physical
 * ring spans all `Tplane * 32` slots so that its shared-memory mapping wraps
 * on a multiple of the warp size. The kernel checks this capacity before
 * creating the structure.
 *
 * @tparam T Input and output element type.
 * @tparam Tidx Rank storage type.
 * @tparam Tplane Number of ring slots held by each warp lane.
 */
template <typename T, typename Tidx = uint8_t, int Tplane = 2>
struct MedianRankWarp {
  static constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
  static constexpr int WARP_SIZE = 32;
  static constexpr int RING_CAPACITY = Tplane * WARP_SIZE;

  Tidx* s_rank;
  uint16_t head;

  /**
   * @brief Initializes ranks for the first window and returns its median.
   *
   * @param s_window Pointer to the first element of the shared-memory window.
   * @param s_rankStorage Pointer to this warp's shared rank storage.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value of the initialized window.
   */
  __device__ T initialize(const T* s_window, Tidx* s_rankStorage,
                          const uint16_t windowLen) {
    const int lane = threadIdx.x;
    s_rank = s_rankStorage;
    head = 0;

    for (int plane = 0; plane < Tplane; ++plane) {
      const int ringSlot = plane * WARP_SIZE + lane;
      const bool valid = ringSlot < windowLen;
      const T laneValue = valid ? s_window[ringSlot] : T{};
      Tidx laneRank = 0;

      for (int sourceSlot = 0; sourceSlot < windowLen; ++sourceSlot) {
        const int sourceLane = sourceSlot % WARP_SIZE;
        T sourceValue{};
        if (lane == sourceLane)
          sourceValue = s_window[sourceSlot];
        sourceValue = __shfl_sync(FULL_WARP_MASK, sourceValue, sourceLane);

        if (valid && (sourceValue < laneValue ||
                      (sourceValue == laneValue && sourceSlot < ringSlot)))
          ++laneRank;
      }

      s_rank[ringSlot] = laneRank;
    }

    return getMedian(s_window, windowLen);
  }

  /**
   * @brief Advances the rank structure by one element and returns the median.
   *
   * @details `s_window` advances by one element on every call. `head` advances
   * across all `Tplane * 32` physical ring slots, so unused slots after the
   * logical window become incoming slots before the ring wraps. Existing
   * values are read directly from shared memory; no values are copied into
   * this structure.
   *
   * @param s_window Pointer to the first element of the new shared-memory
   * window.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value of the updated window.
   */
  __device__ T update(const T* s_window, const uint16_t windowLen) {
    const int lane = threadIdx.x;
    const uint16_t outgoingSlot = head;
    const int outgoingLane = outgoingSlot % WARP_SIZE;

    Tidx outgoingRank = 0;
    if (lane == outgoingLane)
      outgoingRank = s_rank[outgoingSlot];
    outgoingRank = __shfl_sync(FULL_WARP_MASK, outgoingRank, outgoingLane);

    ++head;
    if (head == RING_CAPACITY)
      head = 0;
    int incomingSlot = head + windowLen - 1;
    if (incomingSlot >= RING_CAPACITY)
      incomingSlot -= RING_CAPACITY;
    const int incomingLane = incomingSlot % WARP_SIZE;

    const T incomingValue = s_window[windowLen - 1];
    int incomingRank = 0;

    for (int plane = 0; plane < Tplane; ++plane) {
      const int ringSlot = plane * WARP_SIZE + lane;
      int relative = ringSlot - head;
      if (relative < 0)
        relative += RING_CAPACITY;
      const bool existing = relative < windowLen - 1;
      const T laneValue = existing ? s_window[relative] : T{};

      const bool laneValueIsLess =
          existing && (laneValue < incomingValue ||
                       (laneValue == incomingValue && ringSlot < incomingSlot));
      incomingRank += __popc(__ballot_sync(FULL_WARP_MASK, laneValueIsLess));

      if (existing) {
        Tidx updatedRank = s_rank[ringSlot];
        if (updatedRank > outgoingRank)
          --updatedRank;
        if (incomingValue < laneValue ||
            (incomingValue == laneValue && incomingSlot < ringSlot))
          ++updatedRank;
        s_rank[ringSlot] = updatedRank;
      }
    }

    if (lane == incomingLane)
      s_rank[incomingSlot] = static_cast<Tidx>(incomingRank);

    return getMedian(s_window, windowLen);
  }

private:
  /**
   * @brief Finds and broadcasts the value with the median rank.
   *
   * @param s_window Pointer to the first element of the current shared-memory
   * window.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value, broadcast to every lane in the warp.
   */
  __device__ T getMedian(const T* s_window, const uint16_t windowLen) const {
    const int lane = threadIdx.x;
    const Tidx medianRank = static_cast<Tidx>(windowLen / 2);
    bool laneHasMedian = false;
    T laneMedian{};

    for (int plane = 0; plane < Tplane; ++plane) {
      const int ringSlot = plane * WARP_SIZE + lane;
      int relative = ringSlot - head;
      if (relative < 0)
        relative += RING_CAPACITY;
      if (relative < windowLen && s_rank[ringSlot] == medianRank) {
        laneMedian = s_window[relative];
        laneHasMedian = true;
      }
    }

    const unsigned int medianMask =
        __ballot_sync(FULL_WARP_MASK, laneHasMedian);
    const int medianLane = __ffs(medianMask) - 1;
    return __shfl_sync(FULL_WARP_MASK, laneMedian, medianLane);
  }
};

/**
 * @brief Applies an odd-length sliding median filter independently to rows.
 *
 * @details Each warp processes one contiguous output segment in a row. It
 * first cooperatively loads `outLenPerWarp + windowLen - 1` values into its
 * shared-memory workspace, including the left and right halo. Out-of-range
 * halo elements are assumed to be zero for now. The warp then advances a
 * `MedianRankWarp` through the workspace one output at a time.
 *
 * Consumed workspace elements are overwritten with output medians. Once the
 * segment is complete, the first `outLenPerWarp` workspace elements are copied
 * coalesced to the output image. Input and output images must not alias.
 *
 * The launch must use `dim3(32, warpsPerBlock)` threads and provide
 * `warpsPerBlock * ((outLenPerWarp + windowLen - 1) * sizeof(T) +
 * Tplane * 32 * sizeof(Tidx))` bytes of dynamic shared memory.
 *
 * @tparam T Input and output element type supported by `SharedMemory`.
 * @tparam Tidx Rank storage type.
 * @tparam Tplane Number of ring slots held by each warp lane.
 * @param d_input Input image containing equal-length rows.
 * @param d_output Output image with the same dimensions as `d_input`.
 * @param windowLen Odd median-window length.
 * @param outLenPerWarp Maximum number of output elements produced by a warp.
 */
template <typename T, typename Tidx = uint8_t, int Tplane = 2>
__global__ void rowwise_median_rank_kernel(
    const containers::Image<const T> d_input, containers::Image<T> d_output,
    const uint16_t windowLen, const uint16_t outLenPerWarp) {
  if (blockDim.x != 32 || blockDim.z != 1 || windowLen == 0 ||
      windowLen % 2 == 0 || outLenPerWarp == 0 ||
      windowLen > Tplane * MedianRankWarp<T, Tidx, Tplane>::WARP_SIZE ||
      windowLen - 1 > cuda::std::numeric_limits<Tidx>::max() ||
      d_input.width != d_output.width || d_input.height != d_output.height)
    return;

  const int lane = threadIdx.x;
  const int warpInBlock = threadIdx.y;
  const int workspaceLen = outLenPerWarp + windowLen - 1;
  const int outputStart = blockIdx.x * outLenPerWarp;
  if (outputStart >= d_output.width)
    return;

  SharedMemory<T> s_sharedMemory;
  T* s_blockData = s_sharedMemory.getPointer();
  T* s_warpData = s_blockData + warpInBlock * workspaceLen;
  Tidx* s_blockRank = reinterpret_cast<Tidx*>(
      s_blockData + blockDim.y * workspaceLen);
  Tidx* s_warpRank =
      s_blockRank + warpInBlock * MedianRankWarp<T, Tidx, Tplane>::RING_CAPACITY;
  const int validOutCount = outputStart + outLenPerWarp < d_output.width
                                ? outLenPerWarp
                                : d_output.width - outputStart;
  const int radius = windowLen / 2;

  for (int row = blockIdx.y * blockDim.y + warpInBlock; row < d_output.height;
       row += gridDim.y * blockDim.y) {
    for (int workspaceIndex = lane; workspaceIndex < workspaceLen;
         workspaceIndex += MedianRankWarp<T, Tidx, Tplane>::WARP_SIZE) {
      const int inputCol = outputStart - radius + workspaceIndex;
      // Zero padding is assumed for out-of-range input columns for now.
      s_warpData[workspaceIndex] =
          d_input.colIsValid(inputCol) ? d_input.at(row, inputCol) : T{0};
    }
    __syncwarp();

    MedianRankWarp<T, Tidx, Tplane> d_medianRank;
    for (int outputIndex = 0; outputIndex < validOutCount; ++outputIndex) {
      const T* s_window = s_warpData + outputIndex;
      const T median =
          outputIndex == 0
              ? d_medianRank.initialize(s_window, s_warpRank, windowLen)
              : d_medianRank.update(s_window, windowLen);
      if (lane == 0)
        s_warpData[outputIndex] = median;
    }
    __syncwarp();

    for (int outputIndex = lane; outputIndex < validOutCount;
         outputIndex += MedianRankWarp<T, Tidx, Tplane>::WARP_SIZE)
      d_output.at(row, outputStart + outputIndex) = s_warpData[outputIndex];
    __syncwarp();
  }
}

// =============================================
// =============================================
// ====== COLUMN-WISE 1D MEDIAN FILTER =========
// =============================================
// =============================================

/**
 * @brief Thread-local rank state for a sliding median window.
 *
 * @details One thread owns one vertical column segment. Values remain in fixed
 * ring slots, and `head` identifies the slot replaced by the next incoming
 * value. Ranks use ring-slot order to distinguish duplicate values.
 *
 * @tparam T Input and output element type.
 * @tparam Tidx Rank storage type.
 * @tparam Tcapacity Maximum supported window length.
 */
template <typename T, typename Tidx = uint8_t, int Tcapacity = 64>
struct MedianRankThread {
  T value[Tcapacity];
  Tidx rank[Tcapacity];
  uint16_t head;

  /**
   * @brief Loads and ranks the initial centered window for one column.
   *
   * @details Threads in a block request the same input row at each iteration,
   * so neighboring column reads are globally coalesced. Input rows above or
   * below the image are assumed to contain zero for now.
   *
   * @param d_input Input image.
   * @param col Column owned by this thread.
   * @param outputStart First output row in this thread's segment.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value of the initial window.
   */
  __device__ T initialize(const containers::Image<const T> d_input,
                          const int col, const int outputStart,
                          const uint16_t windowLen) {
    head = 0;
    const int radius = windowLen / 2;
    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      const int inputRow = outputStart - radius + ringSlot;
      value[ringSlot] =
          d_input.rowIsValid(inputRow) ? d_input.at(inputRow, col) : T{0};
    }

    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      Tidx slotRank = 0;
      for (int sourceSlot = 0; sourceSlot < windowLen; ++sourceSlot) {
        if (value[sourceSlot] < value[ringSlot] ||
            (value[sourceSlot] == value[ringSlot] && sourceSlot < ringSlot))
          ++slotRank;
      }
      rank[ringSlot] = slotRank;
    }

    return getMedian(windowLen);
  }

  /**
   * @brief Replaces the outgoing ring value and returns the updated median.
   *
   * @param incomingValue New value entering the final position of the window.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value after the sliding update.
   */
  __device__ T update(const T incomingValue, const uint16_t windowLen) {
    const uint16_t outgoingSlot = head;
    const Tidx outgoingRank = rank[outgoingSlot];
    int incomingRank = 0;

    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      if (ringSlot == outgoingSlot)
        continue;

      if (value[ringSlot] < incomingValue ||
          (value[ringSlot] == incomingValue && ringSlot < outgoingSlot))
        ++incomingRank;

      Tidx updatedRank = rank[ringSlot];
      if (updatedRank > outgoingRank)
        --updatedRank;
      if (incomingValue < value[ringSlot] ||
          (incomingValue == value[ringSlot] && outgoingSlot < ringSlot))
        ++updatedRank;
      rank[ringSlot] = updatedRank;
    }

    value[outgoingSlot] = incomingValue;
    rank[outgoingSlot] = static_cast<Tidx>(incomingRank);
    ++head;
    if (head == windowLen)
      head = 0;

    return getMedian(windowLen);
  }

private:
  /**
   * @brief Returns the value currently carrying the median rank.
   *
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value of the current ring contents.
   */
  __device__ T getMedian(const uint16_t windowLen) const {
    const Tidx medianRank = static_cast<Tidx>(windowLen / 2);
    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      if (rank[ringSlot] == medianRank)
        return value[ringSlot];
    }
    return T{};
  }
};

/**
 * @brief Applies an odd-length sliding median filter independently to columns.
 *
 * @details Each thread processes one contiguous vertical output segment. At
 * every input and output row, neighboring threads access neighboring columns,
 * keeping global reads and writes coalesced without transposing the image.
 * `blockIdx.y` selects the vertical segment.
 *
 * Input rows outside the image are assumed to contain zero for now. Input and
 * output images must not alias. Launch with a one-dimensional thread block
 * containing a multiple of 32 threads and at least 64 threads.
 *
 * @tparam T Input and output element type.
 * @tparam Tidx Rank storage type.
 * @tparam Tcapacity Maximum supported window length.
 * @param d_input Input image containing equal-length columns.
 * @param d_output Output image with the same dimensions as `d_input`.
 * @param windowLen Odd median-window length.
 * @param outLenPerThread Maximum number of output rows produced by one thread.
 */
template <typename T, typename Tidx = uint8_t, int Tcapacity = 64>
__global__ void columnwise_median_rank_kernel(
    const containers::Image<const T> d_input, containers::Image<T> d_output,
    const uint16_t windowLen, const uint16_t outLenPerThread) {
  if (blockDim.x < 64 || blockDim.x % 32 != 0 || blockDim.y != 1 ||
      blockDim.z != 1 || windowLen == 0 || windowLen % 2 == 0 ||
      windowLen > Tcapacity || outLenPerThread == 0 ||
      windowLen - 1 > cuda::std::numeric_limits<Tidx>::max() ||
      d_input.width != d_output.width || d_input.height != d_output.height ||
      d_input.width <= 0 || d_input.height <= 0)
    return;

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int outputStart = blockIdx.y * outLenPerThread;
  if (!d_output.colIsValid(col) || !d_output.rowIsValid(outputStart))
    return;

  const int validOutCount = outputStart + outLenPerThread < d_output.height
                                ? outLenPerThread
                                : d_output.height - outputStart;
  MedianRankThread<T, Tidx, Tcapacity> d_medianRank;
  d_output.at(outputStart, col) =
      d_medianRank.initialize(d_input, col, outputStart, windowLen);

  const int radius = windowLen / 2;
  for (int outputOffset = 1; outputOffset < validOutCount; ++outputOffset) {
    const int outputRow = outputStart + outputOffset;
    const int inputRow = outputRow + radius;
    // Zero padding is assumed for out-of-range input rows for now.
    const T incomingValue =
        d_input.rowIsValid(inputRow) ? d_input.at(inputRow, col) : T{0};
    d_output.at(outputRow, col) = d_medianRank.update(incomingValue, windowLen);
  }
}

// =============================================
// ====== SHARED-MEMORY COLUMN MEDIAN ===========
// =============================================

/**
 * @brief Shared-memory-backed rank state for one column segment.
 *
 * @details One instance exists per thread. Values and ranks occupy slot-major
 * shared-memory arrays owned by the block, while `head` remains thread-local.
 * For a given ring slot, neighboring threads access neighboring shared-memory
 * locations.
 *
 * @tparam T Input and output element type.
 * @tparam Tidx Rank storage type.
 * @tparam Tcapacity Maximum supported window length.
 */
template <typename T, typename Tidx = uint8_t, int Tcapacity = 64>
struct MedianRankThreadShmem {
  T* s_value;
  Tidx* s_rank;
  int stride;
  uint16_t head;

  /**
   * @brief Loads and ranks the initial centered window for one column segment.
   *
   * @param s_valueStorage Shared value storage for the complete block.
   * @param s_rankStorage Shared rank storage for the complete block.
   * @param storageStride Number of threads represented by each ring slot.
   * @param d_input Input image.
   * @param col Column owned by this thread.
   * @param outputStart First output row in this thread's segment.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value of the initial window.
   */
  __device__ T initialize(T* s_valueStorage, Tidx* s_rankStorage,
                          const int storageStride,
                          const containers::Image<const T> d_input,
                          const int col, const int outputStart,
                          const uint16_t windowLen) {
    s_value = s_valueStorage;
    s_rank = s_rankStorage;
    stride = storageStride;
    head = 0;

    const int radius = windowLen / 2;
    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      const int inputRow = outputStart - radius + ringSlot;
      const int storageIndex = ringSlot * stride + threadIdx.x;
      s_value[storageIndex] =
          d_input.rowIsValid(inputRow) ? d_input.at(inputRow, col) : T{0};
    }

    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      const int storageIndex = ringSlot * stride + threadIdx.x;
      Tidx slotRank = 0;
      for (int sourceSlot = 0; sourceSlot < windowLen; ++sourceSlot) {
        const int sourceIndex = sourceSlot * stride + threadIdx.x;
        if (s_value[sourceIndex] < s_value[storageIndex] ||
            (s_value[sourceIndex] == s_value[storageIndex] &&
             sourceSlot < ringSlot))
          ++slotRank;
      }
      s_rank[storageIndex] = slotRank;
    }

    return getMedian(windowLen);
  }

  /**
   * @brief Replaces the outgoing shared ring value and returns the median.
   *
   * @param incomingValue New value entering the final position of the window.
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value after the sliding update.
   */
  __device__ T update(const T incomingValue, const uint16_t windowLen) {
    const uint16_t outgoingSlot = head;
    const int outgoingIndex = outgoingSlot * stride + threadIdx.x;
    const Tidx outgoingRank = s_rank[outgoingIndex];
    int incomingRank = 0;

    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      if (ringSlot == outgoingSlot)
        continue;

      const int storageIndex = ringSlot * stride + threadIdx.x;
      if (s_value[storageIndex] < incomingValue ||
          (s_value[storageIndex] == incomingValue && ringSlot < outgoingSlot))
        ++incomingRank;

      Tidx updatedRank = s_rank[storageIndex];
      if (updatedRank > outgoingRank)
        --updatedRank;
      if (incomingValue < s_value[storageIndex] ||
          (incomingValue == s_value[storageIndex] && outgoingSlot < ringSlot))
        ++updatedRank;
      s_rank[storageIndex] = updatedRank;
    }

    s_value[outgoingIndex] = incomingValue;
    s_rank[outgoingIndex] = static_cast<Tidx>(incomingRank);
    ++head;
    if (head == windowLen)
      head = 0;

    return getMedian(windowLen);
  }

private:
  /**
   * @brief Returns the shared ring value carrying the median rank.
   *
   * @param windowLen Number of elements in the odd-length median window.
   * @return Median value of the current ring contents.
   */
  __device__ T getMedian(const uint16_t windowLen) const {
    const Tidx medianRank = static_cast<Tidx>(windowLen / 2);
    for (int ringSlot = 0; ringSlot < windowLen; ++ringSlot) {
      const int storageIndex = ringSlot * stride + threadIdx.x;
      if (s_rank[storageIndex] == medianRank)
        return s_value[storageIndex];
    }
    return T{};
  }
};

/**
 * @brief Applies the column-wise sliding median using shared ring storage.
 *
 * @details The output mapping and filtering behavior match
 * `columnwise_median_rank_kernel`. The launch must provide
 * `blockDim.x * windowLen * (sizeof(T) + sizeof(Tidx))` bytes of dynamic
 * shared memory. Input and output images must not alias.
 *
 * @tparam T Input and output element type.
 * @tparam Tidx Rank storage type.
 * @tparam Tcapacity Maximum supported window length.
 * @param d_input Input image containing equal-length columns.
 * @param d_output Output image with the same dimensions as `d_input`.
 * @param windowLen Odd median-window length.
 * @param outLenPerThread Maximum number of output rows produced by one thread.
 */
template <typename T, typename Tidx = uint8_t, int Tcapacity = 64>
__global__ void columnwise_median_rank_shmem_kernel(
    const containers::Image<const T> d_input, containers::Image<T> d_output,
    const uint16_t windowLen, const uint16_t outLenPerThread) {
  if (blockDim.x < 64 || blockDim.x % 32 != 0 || blockDim.y != 1 ||
      blockDim.z != 1 || windowLen == 0 || windowLen % 2 == 0 ||
      windowLen > Tcapacity || outLenPerThread == 0 ||
      windowLen - 1 > cuda::std::numeric_limits<Tidx>::max() ||
      d_input.width != d_output.width || d_input.height != d_output.height ||
      d_input.width <= 0 || d_input.height <= 0)
    return;

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int outputStart = blockIdx.y * outLenPerThread;
  if (!d_output.colIsValid(col) || !d_output.rowIsValid(outputStart))
    return;

  extern __shared__ __align__(16) unsigned char s_storage[];
  const size_t slotCount = static_cast<size_t>(blockDim.x) * windowLen;
  T* s_value = reinterpret_cast<T*>(s_storage);
  Tidx* s_rank = reinterpret_cast<Tidx*>(s_value + slotCount);

  const int validOutCount = outputStart + outLenPerThread < d_output.height
                                ? outLenPerThread
                                : d_output.height - outputStart;
  MedianRankThreadShmem<T, Tidx, Tcapacity> d_medianRank;
  d_output.at(outputStart, col) = d_medianRank.initialize(
      s_value, s_rank, blockDim.x, d_input, col, outputStart, windowLen);

  const int radius = windowLen / 2;
  for (int outputOffset = 1; outputOffset < validOutCount; ++outputOffset) {
    const int outputRow = outputStart + outputOffset;
    const int inputRow = outputRow + radius;
    // Zero padding is assumed for out-of-range input rows for now.
    const T incomingValue =
        d_input.rowIsValid(inputRow) ? d_input.at(inputRow, col) : T{0};
    d_output.at(outputRow, col) = d_medianRank.update(incomingValue, windowLen);
  }
}

#include "median.cuh"

#include "block_and_grid_sizing.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <thrust/fill.h>

template <typename T>
std::vector<T> rowwiseMedianCpu(const containers::Image<const T> h_input,
                                const int windowLen) {
  std::vector<T> output(h_input.width * h_input.height);
  std::vector<T> window(windowLen);
  const int radius = windowLen / 2;

  for (int row = 0; row < h_input.height; ++row) {
    for (int col = 0; col < h_input.width; ++col) {
      for (int windowIndex = 0; windowIndex < windowLen; ++windowIndex) {
        const int inputCol = col - radius + windowIndex;
        window[windowIndex] = h_input.colIsValid(inputCol)
                                  ? h_input.at(row, inputCol)
                                  : T{0};
      }
      const auto median = window.begin() + windowLen / 2;
      std::nth_element(window.begin(), median, window.end());
      output[row * h_input.width + col] = *median;
    }
  }

  return output;
}

template <int WARPS_PER_BLOCK, int OUT_LEN_PER_WARP>
void testRowwiseMedianRank(const int numRows) {
  using T = int32_t;
  constexpr int width = 1000;
  constexpr uint16_t windowLen = 55;
  constexpr T outputInitialValue = -9999;

  containers::PinnedHostImageStorage<T> h_input(width, numRows);
  containers::PinnedHostImageStorage<T> h_output(width, numRows);
  containers::DeviceImageStorage<T> d_input(width, numRows);
  containers::DeviceImageStorage<T> d_output(width, numRows);

  auto h_inputImage = h_input.image();
  for (int row = 0; row < numRows; ++row) {
    for (int col = 0; col < width; ++col) {
      const uint32_t index = row * width + col;
      h_inputImage.at(row, col) =
          static_cast<T>((index * 37U + row * 17U) % 17U) - 8;
    }
  }

  h_input.toDevice(d_input);
  thrust::fill(d_output.vec.begin(), d_output.vec.end(), outputInitialValue);

  constexpr dim3 threads(32, WARPS_PER_BLOCK);
  const dim3 blocks(
      justEnoughBlocks(static_cast<unsigned int>(OUT_LEN_PER_WARP),
                       static_cast<unsigned int>(width)),
      justEnoughBlocks(threads.y, static_cast<unsigned int>(numRows)));
  constexpr size_t sharedMemoryBytes =
      WARPS_PER_BLOCK * (OUT_LEN_PER_WARP + windowLen - 1) * sizeof(T);

  rowwise_median_rank_kernel<T, uint8_t, 2>
      <<<blocks, threads, sharedMemoryBytes>>>(
          d_input.cimage(), d_output.image(), windowLen, OUT_LEN_PER_WARP);
  d_output.toHost(h_output);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  const std::vector<T> expected =
      rowwiseMedianCpu(h_input.cimage(), windowLen);
  const auto h_outputImage = h_output.cimage();
  for (int row = 0; row < numRows; ++row) {
    for (int col = 0; col < width; ++col) {
      EXPECT_EQ(h_outputImage.at(row, col), expected[row * width + col])
          << "row " << row << ", col " << col;
    }
  }
}

TEST(RowwiseMedianRank, OneRowOut32) { testRowwiseMedianRank<1, 32>(1); }

TEST(RowwiseMedianRank, OneRowOut64) { testRowwiseMedianRank<1, 64>(1); }

TEST(RowwiseMedianRank, FourRowsOut32) { testRowwiseMedianRank<4, 32>(4); }

TEST(RowwiseMedianRank, FourRowsOut64) { testRowwiseMedianRank<4, 64>(4); }

// =============================================
// =============================================
// ====== COLUMN-WISE 1D MEDIAN FILTER =========
// =============================================
// =============================================

template <typename T>
std::vector<T> columnwiseMedianCpu(const containers::Image<const T> h_input,
                                   const int windowLen) {
  std::vector<T> output(h_input.width * h_input.height);
  std::vector<T> window(windowLen);
  const int radius = windowLen / 2;

  for (int col = 0; col < h_input.width; ++col) {
    for (int row = 0; row < h_input.height; ++row) {
      for (int windowIndex = 0; windowIndex < windowLen; ++windowIndex) {
        const int inputRow = row - radius + windowIndex;
        window[windowIndex] = h_input.rowIsValid(inputRow)
                                  ? h_input.at(inputRow, col)
                                  : T{0};
      }
      const auto median = window.begin() + windowLen / 2;
      std::nth_element(window.begin(), median, window.end());
      output[row * h_input.width + col] = *median;
    }
  }

  return output;
}

template <int THREADS_PER_BLOCK, int OUT_LEN_PER_THREAD,
          bool USE_SHARED_MEMORY = false>
void testColumnwiseMedianRank(const int numCols) {
  using T = int32_t;
  constexpr int height = 1000;
  constexpr uint16_t windowLen = 55;
  constexpr T outputInitialValue = -9999;

  containers::PinnedHostImageStorage<T> h_input(numCols, height);
  containers::PinnedHostImageStorage<T> h_output(numCols, height);
  containers::DeviceImageStorage<T> d_input(numCols, height);
  containers::DeviceImageStorage<T> d_output(numCols, height);

  auto h_inputImage = h_input.image();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < numCols; ++col) {
      const uint32_t index = row * numCols + col;
      h_inputImage.at(row, col) =
          static_cast<T>((index * 37U + col * 11U) % 17U) - 8;
    }
  }

  h_input.toDevice(d_input);
  thrust::fill(d_output.vec.begin(), d_output.vec.end(), outputInitialValue);

  constexpr int threads = THREADS_PER_BLOCK;
  const dim3 blocks(justEnoughBlocks(threads, numCols),
                    justEnoughBlocks(OUT_LEN_PER_THREAD, height));
  if constexpr (USE_SHARED_MEMORY) {
    constexpr size_t sharedMemoryBytes =
        THREADS_PER_BLOCK * windowLen * (sizeof(T) + sizeof(uint8_t));
    columnwise_median_rank_shmem_kernel<T, uint8_t, 64>
        <<<blocks, threads, sharedMemoryBytes>>>(
            d_input.cimage(), d_output.image(), windowLen, OUT_LEN_PER_THREAD);
  } else {
    columnwise_median_rank_kernel<T, uint8_t, 64><<<blocks, threads>>>(
        d_input.cimage(), d_output.image(), windowLen, OUT_LEN_PER_THREAD);
  }
  d_output.toHost(h_output);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  const std::vector<T> expected =
      columnwiseMedianCpu(h_input.cimage(), windowLen);
  const auto h_outputImage = h_output.cimage();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < numCols; ++col) {
      EXPECT_EQ(h_outputImage.at(row, col), expected[row * numCols + col])
          << "row " << row << ", col " << col;
    }
  }
}

TEST(ColumnwiseMedianRank, SixtyFourColumnsThreads64Out32) {
  testColumnwiseMedianRank<64, 32>(64);
}

TEST(ColumnwiseMedianRank, OneHundredColumnsThreads64Out64) {
  testColumnwiseMedianRank<64, 64>(100);
}

TEST(ColumnwiseMedianRank, OneTwentyEightColumnsThreads128Out32) {
  testColumnwiseMedianRank<128, 32>(128);
}

TEST(ColumnwiseMedianRank, TwoFiftySevenColumnsThreads128Out64) {
  testColumnwiseMedianRank<128, 64>(257);
}

TEST(ColumnwiseMedianRankShmem, SixtyFourColumnsThreads64Out32) {
  testColumnwiseMedianRank<64, 32, true>(64);
}

TEST(ColumnwiseMedianRankShmem, OneHundredColumnsThreads64Out64) {
  testColumnwiseMedianRank<64, 64, true>(100);
}

TEST(ColumnwiseMedianRankShmem, OneTwentyEightColumnsThreads128Out32) {
  testColumnwiseMedianRank<128, 32, true>(128);
}

TEST(ColumnwiseMedianRankShmem, TwoFiftySevenColumnsThreads128Out64) {
  testColumnwiseMedianRank<128, 64, true>(257);
}

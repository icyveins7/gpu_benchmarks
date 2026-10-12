#include "median2d.cuh"

#include "block_and_grid_sizing.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <thrust/fill.h>

template <typename T>
std::vector<T> median2dCpu(const containers::Image<const T> input,
                           const uint2 windowDims) {
  std::vector<T> output(input.width * input.height);
  std::vector<T> window;
  window.reserve(windowDims.x * windowDims.y);

  const int halfWindowWidth = windowDims.x / 2;
  const int halfWindowHeight = windowDims.y / 2;
  for (int outputRow = 0; outputRow < input.height; ++outputRow) {
    for (int outputCol = 0; outputCol < input.width; ++outputCol) {
      window.clear();
      for (unsigned int windowRow = 0; windowRow < windowDims.y; ++windowRow) {
        const int inputRow =
            outputRow - halfWindowHeight + static_cast<int>(windowRow);
        for (unsigned int windowCol = 0; windowCol < windowDims.x;
             ++windowCol) {
          const int inputCol =
              outputCol - halfWindowWidth + static_cast<int>(windowCol);
          window.push_back(input.rowIsValid(inputRow) &&
                                   input.colIsValid(inputCol)
                               ? input.at(inputRow, inputCol)
                               : T{0});
        }
      }

      const auto median = window.begin() + window.size() / 2;
      std::nth_element(window.begin(), median, window.end());
      output[outputRow * input.width + outputCol] = *median;
    }
  }

  return output;
}

TEST(Median2d, MatchesNthElementWithZeroPadding) {
  using T = int32_t;
  constexpr int width = 7;
  constexpr int height = 5;
  constexpr dim3 tpb(32, 4);
  constexpr T outputInitialValue = -9999;

  containers::PinnedHostImageStorage<T> hostInput(width, height);
  containers::PinnedHostImageStorage<T> hostOutput(width, height);
  containers::DeviceImageStorage<T> deviceInput(width, height);
  containers::DeviceImageStorage<T> deviceOutput(width, height);
  auto input = hostInput.image();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col)
      input.at(row, col) = (row * width + col) * 37 % 101 - 50;
  }
  hostInput.toDevice(deviceInput);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  const dim3 blocks(justEnoughBlocks(tpb.x, static_cast<unsigned int>(width)),
                    justEnoughBlocks(tpb.y, static_cast<unsigned int>(height)));
  for (const uint2 windowDims : {uint2{1, 1}, uint2{3, 3}, uint2{5, 3}}) {
    thrust::fill(deviceOutput.vec.begin(), deviceOutput.vec.end(),
                 outputInitialValue);
    const size_t sharedMemoryBytes =
        tpb.x * tpb.y * windowDims.x * windowDims.y * sizeof(T);
    median2d_naive_kernel<T><<<blocks, tpb, sharedMemoryBytes>>>(
        deviceInput.cimage(), deviceOutput.image(), windowDims);
    deviceOutput.toHost(hostOutput);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    const std::vector<T> expected = median2dCpu(hostInput.cimage(), windowDims);
    const auto output = hostOutput.cimage();
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        EXPECT_EQ(output.at(row, col), expected[row * width + col])
            << "row " << row << ", col " << col << ", window "
            << windowDims.x << 'x' << windowDims.y;
      }
    }
  }
}

TEST(Median2d, InvalidWindowDimensionsLeaveOutputUnchanged) {
  using T = int32_t;
  constexpr int width = 3;
  constexpr int height = 3;
  constexpr dim3 tpb(32, 4);
  constexpr T outputInitialValue = -1;

  containers::PinnedHostImageStorage<T> hostOutput(width, height);
  containers::DeviceImageStorage<T> deviceInput(width, height);
  containers::DeviceImageStorage<T> deviceOutput(width, height);
  thrust::fill(deviceInput.vec.begin(), deviceInput.vec.end(), T{1});
  thrust::fill(deviceOutput.vec.begin(), deviceOutput.vec.end(),
               outputInitialValue);
  const dim3 blocks(justEnoughBlocks(tpb.x, static_cast<unsigned int>(width)),
                    justEnoughBlocks(tpb.y, static_cast<unsigned int>(height)));

  for (const uint2 windowDims : {uint2{0, 3}, uint2{3, 0}, uint2{2, 3},
                                 uint2{3, 2}}) {
    const size_t sharedMemoryBytes =
        tpb.x * tpb.y * windowDims.x * windowDims.y * sizeof(T);
    median2d_naive_kernel<T><<<blocks, tpb, sharedMemoryBytes>>>(
        deviceInput.cimage(), deviceOutput.image(), windowDims);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  }

  deviceOutput.toHost(hostOutput);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  const auto output = hostOutput.cimage();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col)
      EXPECT_EQ(output.at(row, col), outputInitialValue);
  }
}

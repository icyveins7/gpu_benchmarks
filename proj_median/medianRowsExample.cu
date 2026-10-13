#include "median.cuh"

#include "block_and_grid_sizing.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct MedianRowsSampleValue {
  __host__ __device__ int32_t operator()(const size_t index) const {
    uint32_t value = static_cast<uint32_t>(index);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return static_cast<int32_t>(value);
  }
};

int main(int argc, char *argv[]) {
  if (argc > 7) {
    std::cerr
        << "Usage: " << argv[0]
        << " [rows] [cols] [windowLen] [outLenPerWarp] [warpsPerBlock] [iter]\n";
    return 1;
  }

  int rows = 4;
  int cols = 1000;
  int windowLen = 55;
  int outLenPerWarp = 32;
  int warpsPerBlock = 4;
  int iter = 2;
  if (argc > 1)
    rows = std::atoi(argv[1]);
  if (argc > 2)
    cols = std::atoi(argv[2]);
  if (argc > 3)
    windowLen = std::atoi(argv[3]);
  if (argc > 4)
    outLenPerWarp = std::atoi(argv[4]);
  if (argc > 5)
    warpsPerBlock = std::atoi(argv[5]);
  if (argc > 6)
    iter = std::atoi(argv[6]);

  if (rows <= 0 || cols <= 0 || windowLen <= 0 || windowLen % 2 == 0 ||
      windowLen > 64 || outLenPerWarp <= 0 ||
      outLenPerWarp > std::numeric_limits<uint16_t>::max() ||
      warpsPerBlock <= 0 || warpsPerBlock > 32 || iter <= 0) {
    std::cerr << "Rows, columns, and iterations must be positive; windowLen "
                 "must be positive, odd, and no greater than 64; "
                 "outLenPerWarp must fit in uint16_t; and warpsPerBlock must "
                 "be between 1 and 32.\n";
    return 1;
  }

  using T = int32_t;
  containers::DeviceImageStorage<T> d_input(cols, rows);
  containers::DeviceImageStorage<T> d_output(cols, rows);
  const auto firstIndex = thrust::make_counting_iterator<size_t>(0);
  thrust::transform(firstIndex, firstIndex + d_input.vec.size(),
                    d_input.vec.begin(), MedianRowsSampleValue{});

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    return 1;
  }

  const dim3 threads(32, warpsPerBlock);
  const dim3 blocks(
      justEnoughBlocks(static_cast<unsigned int>(outLenPerWarp),
                       static_cast<unsigned int>(cols)),
      justEnoughBlocks(threads.y, static_cast<unsigned int>(rows)));
  const size_t sharedMemoryBytes =
      static_cast<size_t>(warpsPerBlock) *
      (outLenPerWarp + windowLen - 1) * sizeof(T);

  int device = 0;
  int maxSharedMemory = 0;
  error = cudaGetDevice(&device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(
        &maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    return 1;
  }
  if (sharedMemoryBytes > static_cast<size_t>(maxSharedMemory)) {
    std::cerr << "Requested dynamic shared memory exceeds the device limit of "
              << maxSharedMemory << " bytes.\n";
    return 1;
  }

  error = cudaFuncSetAttribute(
      rowwise_median_rank_kernel<T, uint8_t, 2>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sharedMemoryBytes));
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    return 1;
  }

  std::cout << "Input: " << rows << 'x' << cols << '\n'
            << "Window length: " << windowLen << '\n'
            << "Output length per warp: " << outLenPerWarp << '\n'
            << "Warps per block: " << warpsPerBlock << '\n'
            << "Blocks: " << blocks.x << 'x' << blocks.y << '\n'
            << "Dynamic shared memory: " << sharedMemoryBytes << " bytes\n"
            << "Kernel launches: " << iter << '\n';

  for (int iteration = 0; iteration < iter; ++iteration) {
    rowwise_median_rank_kernel<T, uint8_t, 2>
        <<<blocks, threads, sharedMemoryBytes>>>(
            d_input.cimage(), d_output.image(),
            static_cast<uint16_t>(windowLen),
            static_cast<uint16_t>(outLenPerWarp));
    error = cudaGetLastError();
    if (error == cudaSuccess)
      error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      std::cerr << cudaGetErrorString(error) << '\n';
      return 1;
    }
  }

  return 0;
}

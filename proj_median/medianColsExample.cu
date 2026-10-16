#include "median.cuh"

#include "block_and_grid_sizing.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct MedianColsSampleValue {
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
    std::cerr << "Usage: " << argv[0]
              << " [rows] [cols] [windowLen] [outLenPerThread] "
                 "[threadsPerBlock] [iter]\n";
    return 1;
  }

  int rows = 1000;
  int cols = 64;
  int windowLen = 55;
  int outLenPerThread = 32;
  int threadsPerBlock = 64;
  int iter = 2;
  if (argc > 1)
    rows = std::atoi(argv[1]);
  if (argc > 2)
    cols = std::atoi(argv[2]);
  if (argc > 3)
    windowLen = std::atoi(argv[3]);
  if (argc > 4)
    outLenPerThread = std::atoi(argv[4]);
  if (argc > 5)
    threadsPerBlock = std::atoi(argv[5]);
  if (argc > 6)
    iter = std::atoi(argv[6]);

  if (rows <= 0 || cols <= 0 || windowLen <= 0 || windowLen % 2 == 0 ||
      windowLen > 64 || outLenPerThread <= 0 ||
      outLenPerThread > std::numeric_limits<uint16_t>::max() ||
      threadsPerBlock < 64 || threadsPerBlock > 1024 ||
      threadsPerBlock % 32 != 0 || iter <= 0) {
    std::cerr << "Rows, columns, and iterations must be positive; windowLen "
                 "must be positive, odd, and no greater than 64; "
                 "outLenPerThread must fit in uint16_t; and threadsPerBlock "
                 "must be a multiple of 32 between 64 and 1024.\n";
    return 1;
  }

  using T = int32_t;
  containers::DeviceImageStorage<T> d_input(cols, rows);
  containers::DeviceImageStorage<T> d_output(cols, rows);
  const auto firstIndex = thrust::make_counting_iterator<size_t>(0);
  thrust::transform(firstIndex, firstIndex + d_input.vec.size(),
                    d_input.vec.begin(), MedianColsSampleValue{});

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    return 1;
  }

  const dim3 blocks(
      justEnoughBlocks(static_cast<unsigned int>(threadsPerBlock),
                       static_cast<unsigned int>(cols)),
      justEnoughBlocks(static_cast<unsigned int>(outLenPerThread),
                       static_cast<unsigned int>(rows)));

#ifdef MEDIAN_COLS_USE_SHMEM
  const size_t sharedMemoryBytes =
      static_cast<size_t>(threadsPerBlock) * windowLen *
      (sizeof(T) + sizeof(uint8_t));
  int device = 0;
  int maxSharedMemory = 0;
  error = cudaGetDevice(&device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(
        &maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (error == cudaSuccess &&
      sharedMemoryBytes > static_cast<size_t>(maxSharedMemory)) {
    std::cerr << "Requested dynamic shared memory exceeds the device limit of "
              << maxSharedMemory << " bytes.\n";
    return 1;
  }
  if (error == cudaSuccess)
    error = cudaFuncSetAttribute(
        columnwise_median_rank_shmem_kernel<T, uint8_t, 64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sharedMemoryBytes));
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    return 1;
  }
#endif

  std::cout << "Implementation: "
#ifdef MEDIAN_COLS_USE_SHMEM
            << "shared memory\n"
#else
            << "local memory\n"
#endif
            << "Input: " << rows << 'x' << cols << '\n'
            << "Window length: " << windowLen << '\n'
            << "Output length per thread: " << outLenPerThread << '\n'
            << "Threads per block: " << threadsPerBlock << '\n'
            << "Blocks: " << blocks.x << 'x' << blocks.y << '\n'
            << "Kernel launches: " << iter << '\n';
#ifdef MEDIAN_COLS_USE_SHMEM
  std::cout << "Dynamic shared memory: " << sharedMemoryBytes << " bytes\n";
#endif

  for (int iteration = 0; iteration < iter; ++iteration) {
#ifdef MEDIAN_COLS_USE_SHMEM
    columnwise_median_rank_shmem_kernel<T, uint8_t, 64>
        <<<blocks, threadsPerBlock, sharedMemoryBytes>>>(
            d_input.cimage(), d_output.image(),
            static_cast<uint16_t>(windowLen),
            static_cast<uint16_t>(outLenPerThread));
#else
    columnwise_median_rank_kernel<T, uint8_t, 64>
        <<<blocks, threadsPerBlock>>>(
            d_input.cimage(), d_output.image(),
            static_cast<uint16_t>(windowLen),
            static_cast<uint16_t>(outLenPerThread));
#endif
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

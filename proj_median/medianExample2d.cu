#include "median2d.cuh"

#include "block_and_grid_sizing.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct PseudorandomValue {
  __host__ __device__ int32_t operator()(uint32_t value) const {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return static_cast<int32_t>(value);
  }
};

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input-length> <window-length>\n";
    return 1;
  }

  const int inputLength = std::atoi(argv[1]);
  const int windowLength = std::atoi(argv[2]);
  if (inputLength <= 0 || windowLength <= 0 || windowLength % 2 == 0) {
    std::cerr << "Input length must be positive and window length must be "
                 "positive and odd.\n";
    return 1;
  }

  using T = int32_t;
  containers::DeviceImageStorage<T> input(inputLength, inputLength);
  containers::DeviceImageStorage<T> output(inputLength, inputLength);
  const auto firstIndex = thrust::make_counting_iterator<uint32_t>(0);
  thrust::transform(firstIndex, firstIndex + input.vec.size(), input.vec.begin(),
                    PseudorandomValue{});
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    return 1;
  }

  constexpr dim3 tpb(32, 4);
  const dim3 blocks(
      justEnoughBlocks(tpb.x, static_cast<unsigned int>(inputLength)),
      justEnoughBlocks(tpb.y, static_cast<unsigned int>(inputLength)));
  const uint2 windowDims{static_cast<unsigned int>(windowLength),
                         static_cast<unsigned int>(windowLength)};
  const size_t sharedMemoryBytes = static_cast<size_t>(tpb.x) * tpb.y *
                                   windowDims.x * windowDims.y * sizeof(T);

  std::cout << "Input: " << inputLength << 'x' << inputLength << '\n'
            << "Window: " << windowLength << 'x' << windowLength << '\n'
            << "Blocks: " << blocks.x << 'x' << blocks.y << '\n'
            << "Threads per block: " << tpb.x << 'x' << tpb.y << '\n'
            << "Dynamic shared memory: " << sharedMemoryBytes << " bytes\n"
            << "Kernel launches: 2\n";

  for (int launch = 0; launch < 2; ++launch) {
    median2d_naive_kernel<T><<<blocks, tpb, sharedMemoryBytes>>>(
        input.cimage(), output.image(), windowDims);
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

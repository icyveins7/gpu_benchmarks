/*
Continuous-capture experiment, run from the repository root:

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output='/tmp/profile_sections_2s_%n' ./build/proj_basicops/profile_sections_example 2

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output='/tmp/profile_sections_20s_%n' ./build/proj_basicops/profile_sections_example 20
*/

/*
This repeated-capture command was tried but failed with "Connection to Agent lost"
after the first range on Nsight Systems 2026.1.3. The repeat:3:defer and
repeat-shutdown:3:defer variants failed in the same way.

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=repeat --output='/tmp/profile_sections_repeat_%n' ./build/proj_basicops/profile_sections_example
*/

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstdlib>
#include <thread>

__global__ void increment(int *data, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    ++data[index];
  }
}

int main(int argc, char *argv[]) {
  constexpr int count = 1 << 20;
  constexpr int iterations = 3;
  constexpr int launchesPerSection = 100;
  int gapSeconds = 2;
  if (argc > 1) {
    gapSeconds = std::atoi(argv[1]);
  }

  int *data;
  cudaMalloc(&data, count * sizeof(int));
  cudaMemset(data, 0, count * sizeof(int));
  increment<<<count / 256, 256>>>(data, count);
  cudaDeviceSynchronize();

  cudaProfilerStart();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::this_thread::sleep_for(std::chrono::seconds(gapSeconds));

    nvtxRangePushA("CUDA section");
    for (int launch = 0; launch < launchesPerSection; ++launch) {
      increment<<<count / 256, 256>>>(data, count);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
  cudaProfilerStop();

  /*
  Failed repeated-capture version:

  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::this_thread::sleep_for(std::chrono::seconds(gapSeconds));

    cudaProfilerStart();
    nvtxRangePushA("CUDA section");
    for (int launch = 0; launch < launchesPerSection; ++launch) {
      increment<<<count / 256, 256>>>(data, count);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
    cudaProfilerStop();
  }
  */

  cudaFree(data);
  return 0;
}

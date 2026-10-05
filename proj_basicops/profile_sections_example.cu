/*
See nsys_profile_sections.md for experiment details, results, and limitations.
*/

/*
Continuous-capture experiment, run from the repository root:

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output='./build/profile_sections_2s_%n' \
  ./build/proj_basicops/profile_sections_example 2

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output='./build/profile_sections_20s_%n' \
  ./build/proj_basicops/profile_sections_example 20
*/

/*
NVTX-triggered repeated capture. IMPORTANT: Reconnection required sudo in this
environment!

sudo nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx --nvtx-capture='CUDA section@profile_sections' \
  --capture-range-end=repeat --output='./build/profile_sections_nvtx_%n' \
  ./build/proj_basicops/profile_sections_example_nvtx
*/

/*
CUDA Profiler API repeated capture. IMPORTANT: Reconnection required sudo in
this environment!

sudo nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output='./build/profile_sections_cudaProfilerApi_%n' \
  ./build/proj_basicops/profile_sections_example_cudaProfilerApi
*/

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstdlib>
#include <thread>

#if (defined(PROFILE_SECTIONS_CONTINUOUS) + defined(PROFILE_SECTIONS_NVTX) +   \
     defined(PROFILE_SECTIONS_CUDA_PROFILER_API)) != 1
#error "Define exactly one profile-sections capture mode"
#endif

__global__ void increment(int* data, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    ++data[index];
  }
}

void runCudaSection(int* data, int count, int launches,
                    nvtxDomainHandle_t domain,
                    const nvtxEventAttributes_t* attributes) {
  nvtxDomainRangePushEx(domain, attributes);
  for (int launch = 0; launch < launches; ++launch) {
    increment<<<count / 256, 256>>>(data, count);
  }
  cudaDeviceSynchronize();
  nvtxDomainRangePop(domain);
}

int main(int argc, char* argv[]) {
  constexpr int count = 1 << 20;
  constexpr int iterations = 3;
  constexpr int launchesPerSection = 100;
  int gapSeconds = 2;
  if (argc > 1) {
    gapSeconds = std::atoi(argv[1]);
  }

  int* data;
  cudaMalloc(&data, count * sizeof(int));
  cudaMemset(data, 0, count * sizeof(int));
  increment<<<count / 256, 256>>>(data, count);
  cudaDeviceSynchronize();

  nvtxDomainHandle_t domain = nvtxDomainCreateA("profile_sections");
  nvtxStringHandle_t rangeName =
      nvtxDomainRegisterStringA(domain, "CUDA section");
  nvtxEventAttributes_t attributes{};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attributes.message.registered = rangeName;

#if defined(PROFILE_SECTIONS_CONTINUOUS)
  cudaProfilerStart();
#endif

  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::this_thread::sleep_for(std::chrono::seconds(gapSeconds));

#if defined(PROFILE_SECTIONS_CUDA_PROFILER_API)
    cudaProfilerStart();
#endif

    runCudaSection(data, count, launchesPerSection, domain, &attributes);

#if defined(PROFILE_SECTIONS_CUDA_PROFILER_API)
    cudaProfilerStop();
#endif
  }

#if defined(PROFILE_SECTIONS_CONTINUOUS)
  cudaProfilerStop();
#endif

  nvtxDomainDestroy(domain);
  cudaFree(data);
  return 0;
}

// Example: multiple threads sharing a per-GPU resource via
// containers::PerGpuSingleton, where the resource is a custom struct holding
// both a pinned host buffer and a device buffer (rather than a single
// container, to show that T can be anything you define).
//
// Each worker thread "claims" the resource for a target GPU ID, which either
// constructs it (on the first claim for that GPU) or blocks until whichever
// thread currently holds it releases it. While held, the worker resizes and
// fills the host buffer, copies it to the device buffer, and reduces it on
// the device -- standing in for e.g. staging + uploading + processing data
// through a shared per-GPU workspace. This is repeated across every GPU
// available on the machine, with multiple worker threads sharing each GPU's
// resource.
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "containers/gpu_singleton.cuh"
#include "pinnedalloc.cuh"

// Custom per-GPU resource: a host staging buffer plus a device buffer.
struct GpuBuffers {
  thrust::pinned_host_vector<float> host;
  thrust::device_vector<float> device;
};

using GpuBuffersRegistry = containers::PerGpuSingleton<GpuBuffers>;

void worker(int workerId, int gpuId, size_t len) {
  printf("worker %d waiting to claim GPU %d's buffers\n", workerId, gpuId);
  GpuBuffersRegistry::Handle handle = GpuBuffersRegistry::instance().claim(gpuId);
  printf("worker %d claimed GPU %d's buffers\n", workerId, gpuId);

  handle->host.resize(len);
  for (size_t i = 0; i < len; ++i) {
    handle->host[i] = static_cast<float>(workerId);
  }

  handle->device = handle->host; // H2D copy via thrust's iterator-based copy
  float sum = thrust::reduce(handle->device.begin(), handle->device.end());

  // Hold the resource for a bit so the serialization enforced by the
  // Handle's lock is obvious in the interleaved output above/below.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  printf("worker %d releasing GPU %d's buffers (size %zu, device sum %f)\n",
         workerId, gpuId, handle->host.size(), sum);
}

int main() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Found %d GPU(s)\n", deviceCount);

  const int workersPerGpu = 4;

  std::vector<std::thread> threads;
  for (int gpuId = 0; gpuId < deviceCount; ++gpuId) {
    for (int i = 0; i < workersPerGpu; ++i) {
      threads.emplace_back(worker, i, gpuId, 1024 + i * 256);
    }
  }
  for (auto &t : threads) {
    t.join();
  }

  return 0;
}

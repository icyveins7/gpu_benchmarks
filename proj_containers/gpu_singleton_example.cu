// Example: multiple threads sharing a single pinned_host_vector per GPU via
// containers::PerGpuSingleton.
//
// Each worker thread "claims" the pinned_host_vector for a target GPU ID,
// which either constructs it (on the first claim for that GPU) or blocks
// until whichever thread currently holds it releases it. While held, the
// worker resizes the buffer to fit its own workload and fills it, standing in
// for e.g. staging data into a shared pinned buffer before a H2D copy.
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "containers/gpu_singleton.cuh"
#include "pinnedalloc.cuh"

using PinnedVectorRegistry = containers::PerGpuSingleton<thrust::pinned_host_vector<float>>;

void worker(int workerId, int gpuId, size_t len) {
  printf("worker %d waiting to claim GPU %d's pinned buffer\n", workerId, gpuId);
  PinnedVectorRegistry::Handle handle = PinnedVectorRegistry::instance().claim(gpuId);
  printf("worker %d claimed GPU %d's pinned buffer\n", workerId, gpuId);

  handle->resize(len);
  for (size_t i = 0; i < len; ++i) {
    (*handle)[i] = static_cast<float>(workerId);
  }

  // Hold the resource for a bit so the serialization enforced by the
  // Handle's lock is obvious in the interleaved output above/below.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  printf("worker %d releasing GPU %d's pinned buffer (size %zu, filled with "
         "%f)\n",
         workerId, gpuId, handle->size(), (*handle)[0]);
}

int main() {
  const int gpuId = 0;
  const int numWorkers = 4;

  std::vector<std::thread> threads;
  for (int i = 0; i < numWorkers; ++i) {
    threads.emplace_back(worker, i, gpuId, 1024 + i * 256);
  }
  for (auto &t : threads) {
    t.join();
  }

  return 0;
}

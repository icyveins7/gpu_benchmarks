// Compares four ways of zeroing a large pinned host buffer that is reused
// every iteration:
//   1. std::memset() directly on the pinned host memory from one CPU thread.
//   1a. Four CPU threads, each applying std::memset() to one contiguous
//       quarter.
//   2. cudaMemsetAsync() on a device buffer, then copy it down to the pinned
//      host buffer over PCIe (DeviceImageStorage::toHost).
//   3. cudaMemsetAsync() directly on the pinned buffer's mapped device
//      pointer.
//
// Usage: pinned_host_zeroing [width] [height] [iters]
//
// ---------------------------------------------------------------------------
// Investigation notes (measured on a 1-socket, 80-core Intel Xeon 6781P
// "Granite Rapids" server, 2 NUMA nodes, one GPU per node (L40S), powersave
// governor):
//
// Sweeping buffer size for approach 1 (single-thread host memset) gives a
// clearly non-linear GB/s curve, not a straight line:
//
//   0.25 - 2   MiB   ~58 GB/s   (fits in this core's L2, 2 MiB/core)
//   4          MiB   ~30 GB/s   <- cliff #1: spills L2 -> L3
//   8   - 128  MiB   ~30 GB/s   (flat plateau, still L3-resident)
//   192 - 256  MiB   ~26 -> 17 GB/s  <- cliff #2 starts
//   384 - 1024 MiB   ~16 -> 14.6 GB/s (single-core main-memory floor)
//
// Cliff #1 lines up with this core's L2 capacity (2 MiB) as reported by
// `lscpu`. Cliff #2 was initially suspected to be a TLB effect, but decoding
// CPUID leaf 0x18 (`cpuid -1 -l 0x18 -s <n>`) showed the unified L2 STLB is
// only 1024 entries (8-way, 128 sets) for 4 KiB pages, i.e. ~4 MiB of reach --
// far too small to explain a cliff at 128-512 MiB. A pointer-chasing
// microbenchmark (random order, one access per 4 KiB page) confirmed the STLB
// reach empirically: latency jumps from ~10-20 ns to ~40 ns exactly at 4 MiB
// and stays flat afterward. For a *sequential* memset, though, that per-page
// TLB miss cost is trivially hidden behind the ~130 ns it takes to actually
// write a page's worth of bytes, so TLB reach does not explain cliff #2.
// Cliff #2 is better explained by finally spilling this core's realistic
// share of the (80-core-shared, 336 MiB nominal) L3 cache and becoming limited
// by the single core's ability to drive the main-memory path. Forcing the
// buffer onto the NUMA-remote node (`numactl --cpunodebind=1 --membind=0`)
// only added ~18% latency, so NUMA placement is a secondary contributor at
// most, not the main cause. CPU frequency/thermal throttling under sustained
// large memsets remains a plausible secondary contributor but was not
// confirmed, since `perf`/MSR access requires root privileges not available
// in this environment.
//
// Practical upshot for approach 2 (GPU memset + PCIe copy): the PCIe DMA
// write into pinned host memory is issued by the GPU's copy engine, not by
// CPU load/store instructions, so it does not walk this core's TLBs or pay
// the L2/L3 cliffs above. It is bounded mainly by PCIe link bandwidth and GPU
// memory bandwidth instead (modulo Intel DDIO possibly steering the incoming
// writes into LLC on some platforms, which is a DMA-path optimization, not
// equivalent to a CPU store). For small/medium buffers, single-thread host
// memset wins because its cache-resident throughput dwarfs PCIe bandwidth and
// copy latency. Once it reaches its single-core main-memory floor, the
// comparatively flat GPU+copy path wins. On this machine that crossover was
// roughly 150-256 MiB.
//
// Approach 3 removes the temporary device-buffer memset and writes directly
// to mapped pinned host memory over PCIe. On this L40S/UVA configuration,
// cudaHostGetDevicePointer() returned the same address as the host pointer;
// cudaMemsetAsync() accepted that address and the benchmark verified the full
// buffer was zero. The benchmark still uses the mapped device pointer because
// cudaMemsetAsync() formally expects a device pointer and pointer identity is
// not portable across all CUDA configurations.
//
// At 20000x40000 floats (3.2 GB decimal, equivalent in size to a 20000x20000
// double image), 10 iterations measured:
//
//   single-thread memset   234.465 ms   13.65 GB/s
//   four-thread memset      73.454 ms   43.56 GB/s
//   GPU memset + copy      125.912 ms   25.41 GB/s
//   mapped-host memset     121.188 ms   26.41 GB/s
//
// Thread creation and joining are included in the four-thread measurement,
// and the workers are not affinity-pinned. Four threads being 3.19x faster
// than one and 1.65x faster than mapped-host memset confirms that the
// single-thread result is not the socket's aggregate DRAM limit. It is the
// per-core store/main-memory path that limits approach 1. At 64 MiB, four
// threads did not help because startup, scheduling, and cache effects were
// significant relative to the operation.
//
// The same L3 spillover pattern was measured on a 2-socket Intel Xeon Gold
// 6338T server (24 cores/socket, 1.25 MiB L2/core, 36 MiB L3/socket) with an
// NVIDIA L4 on PCIe 4.0 x16. The crossover was roughly 36-40 MiB; at 4096x4096
// floats (64 MiB), host memset took ~4.04 ms versus ~2.75 ms for GPU+copy.
// All crossover points and throughputs are hardware- and buffer-size-specific;
// re-run this benchmark at production sizes rather than assuming them
// elsewhere.
// ---------------------------------------------------------------------------
#include "containers/image.cuh"
#include "containers/streams.cuh"
#include "timer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <thread>

// Returns the duration (in the timer's Tdur units) of the last start()/stop()
// pair without going through report()'s printing.
template <typename Tdur> double lastElapsed(HighResolutionTimer<Tdur>& timer) {
  const auto& m = timer.measurements();
  return std::chrono::duration<double, typename Tdur::period>(m.back().first -
                                                              m.front().first)
      .count();
}

int main(int argc, char* argv[]) {
  int width = 4096;
  int height = 4096;
  int iters = 20;
  if (argc > 1)
    width = atoi(argv[1]);
  if (argc > 2)
    height = atoi(argv[2]);
  if (argc > 3)
    iters = atoi(argv[3]);

  printf("Zeroing a %dx%d float image, %d iterations\n", width, height, iters);

  containers::PinnedHostImageStorage<float> h_img(width, height);
  containers::DeviceImageStorage<float> d_img(width, height);
  containers::CudaStream stream;

  const size_t numBytes = (size_t)width * height * sizeof(float);
  printf("Buffer size: %.3f MiB\n", numBytes / (1024.0 * 1024.0));

  float* hostPtr = h_img.vec.data().get();
  float* mappedPtr = h_img.mappedImage().data;
  printf("Host pointer: %p, mapped device pointer: %p (%s)\n",
         static_cast<void*>(hostPtr), static_cast<void*>(mappedPtr),
         hostPtr == mappedPtr ? "same" : "different");

  auto testMappedMemset = [&](const char* label, float* ptr) {
    std::memset(hostPtr, 0xff, numBytes);
    cudaError_t err = cudaMemsetAsync(ptr, 0, numBytes, stream());
    if (err == cudaSuccess)
      err = cudaStreamSynchronize(stream());
    if (err != cudaSuccess) {
      printf("[%s] rejected: %s\n", label, cudaGetErrorString(err));
      return false;
    }

    const size_t numElements = numBytes / sizeof(float);
    const bool zeroed = std::all_of(hostPtr, hostPtr + numElements,
                                    [](float value) { return value == 0.0f; });
    printf("[%s] accepted; contents %s\n", label,
           zeroed ? "verified zero" : "NOT zero");
    return zeroed;
  };

  const bool hostPtrWorks = testMappedMemset("host pointer", hostPtr);
  const bool mappedPtrWorks =
      hostPtr == mappedPtr
          ? hostPtrWorks
          : testMappedMemset("mapped device pointer", mappedPtr);

  constexpr int numCpuThreads = 4;
  auto threadedMemset = [&] {
    std::thread threads[numCpuThreads];
    for (int i = 0; i < numCpuThreads; ++i) {
      const size_t begin = numBytes * i / numCpuThreads;
      const size_t end = numBytes * (i + 1) / numCpuThreads;
      threads[i] = std::thread([=] {
        std::memset(reinterpret_cast<unsigned char*>(hostPtr) + begin, 0,
                    end - begin);
      });
    }
    for (auto& thread : threads)
      thread.join();
  };

  HighResolutionTimer<std::chrono::microseconds> timer;

  // Warm up both paths once so we don't measure one-off setup costs
  // (e.g. first-touch page faults, driver/context warmup).
  std::memset(h_img.vec.data().get(), 0, numBytes);
  threadedMemset();
  cudaMemsetAsync(d_img.vec.data().get(), 0, numBytes, stream());
  d_img.toHost(h_img, stream());
  stream.sync();

  // Approach 1: memset directly on the pinned host buffer.
  double memsetTotalUs = 0.0;
  double memsetMinUs = std::numeric_limits<double>::max();
  double memsetMaxUs = 0.0;
  for (int i = 0; i < iters; ++i) {
    timer.clear();
    timer.event("start");
    std::memset(h_img.vec.data().get(), 0, numBytes);
    timer.event("end");
    double us = lastElapsed(timer);
    memsetTotalUs += us;
    memsetMinUs = std::min(memsetMinUs, us);
    memsetMaxUs = std::max(memsetMaxUs, us);
  }
  printf("[memset]          avg %.3f us  min %.3f us  max %.3f us\n",
         memsetTotalUs / iters, memsetMinUs, memsetMaxUs);

  // Approach 1a. memset using multiple threads
  double threadedTotalUs = 0.0;
  double threadedMinUs = std::numeric_limits<double>::max();
  double threadedMaxUs = 0.0;
  for (int i = 0; i < iters; ++i) {
    timer.clear();
    timer.event("start");
    threadedMemset();
    timer.event("end");
    double us = lastElapsed(timer);
    threadedTotalUs += us;
    threadedMinUs = std::min(threadedMinUs, us);
    threadedMaxUs = std::max(threadedMaxUs, us);
  }
  printf("[memset x4]       avg %.3f us  min %.3f us  max %.3f us\n",
         threadedTotalUs / iters, threadedMinUs, threadedMaxUs);

  // Approach 2: zero on the device, then copy down over PCIe.
  double gpuZeroTotalUs = 0.0;
  double gpuZeroMinUs = std::numeric_limits<double>::max();
  double gpuZeroMaxUs = 0.0;
  for (int i = 0; i < iters; ++i) {
    timer.clear();
    timer.event("start");
    cudaMemsetAsync(d_img.vec.data().get(), 0, numBytes, stream());
    d_img.toHost(h_img, stream());
    stream.sync();
    timer.event("end");
    double us = lastElapsed(timer);
    gpuZeroTotalUs += us;
    gpuZeroMinUs = std::min(gpuZeroMinUs, us);
    gpuZeroMaxUs = std::max(gpuZeroMaxUs, us);
  }
  printf("[gpu memset+copy] avg %.3f us  min %.3f us  max %.3f us\n",
         gpuZeroTotalUs / iters, gpuZeroMinUs, gpuZeroMaxUs);

  // Approach 3. zero directly using mapped pinned host pointer
  if (mappedPtrWorks) {
    double mappedTotalUs = 0.0;
    double mappedMinUs = std::numeric_limits<double>::max();
    double mappedMaxUs = 0.0;
    for (int i = 0; i < iters; ++i) {
      timer.clear();
      timer.event("start");
      cudaMemsetAsync(mappedPtr, 0, numBytes, stream());
      stream.sync();
      timer.event("end");
      double us = lastElapsed(timer);
      mappedTotalUs += us;
      mappedMinUs = std::min(mappedMinUs, us);
      mappedMaxUs = std::max(mappedMaxUs, us);
    }
    printf("[mapped memset]   avg %.3f us  min %.3f us  max %.3f us\n",
           mappedTotalUs / iters, mappedMinUs, mappedMaxUs);
  }

  return 0;
}

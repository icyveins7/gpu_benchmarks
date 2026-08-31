// Compares two ways of zeroing a large pinned host buffer that is reused
// every iteration:
//   1. std::memset() directly on the pinned host memory.
//   2. cudaMemsetAsync() on a device buffer, then copy it down to the
//      pinned host buffer over PCIe (DeviceImageStorage::toHost).
//
// Usage: pinned_host_zeroing [width] [height] [iters]
//
// ---------------------------------------------------------------------------
// Investigation notes (measured on a 1-socket, 80-core Intel Xeon 6781P
// "Granite Rapids" server, 2 NUMA nodes, one GPU per node (L40S), powersave
// governor):
//
// Sweeping buffer size for approach 1 (host memset) alone gives a clearly
// non-linear GB/s curve, not a straight line:
//
//   0.25 - 2   MiB   ~58 GB/s   (fits in this core's L2, 2 MiB/core)
//   4          MiB   ~30 GB/s   <- cliff #1: spills L2 -> L3
//   8   - 128  MiB   ~30 GB/s   (flat plateau, still L3-resident)
//   192 - 256  MiB   ~26 -> 17 GB/s  <- cliff #2 starts
//   384 - 1024 MiB   ~16 -> 14.6 GB/s (flattens out; steady-state DRAM)
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
// share of the (80-core-shared, 336 MiB nominal) L3 cache into genuine main
// memory bandwidth. Forcing the buffer onto the NUMA-remote node
// (`numactl --cpunodebind=1 --membind=0`) only added ~18% latency, so NUMA
// placement is a secondary contributor at most, not the main cause. CPU
// frequency/thermal throttling under sustained large memsets remains a
// plausible secondary contributor but was not confirmed, since `perf`/MSR
// access requires root privileges not available in this environment.
//
// Practical upshot for approach 2 (GPU memset + PCIe copy): the PCIe DMA
// write into pinned host memory is issued by the GPU's copy engine, not by
// CPU load/store instructions, so it does not walk this core's TLBs or pay
// the L2/L3/DRAM cliffs above -- it is bounded mainly by PCIe link bandwidth
// and GPU memory bandwidth instead (modulo Intel DDIO possibly steering the
// incoming writes into LLC on some platforms, which is a DMA-path
// optimization, not equivalent to a CPU store). That is consistent with what
// this benchmark shows: for small/medium buffers, host memset comfortably
// wins (its cache-resident throughput dwarfs PCIe bandwidth + copy latency);
// but once the buffer is large enough that host memset has already fallen to
// its DRAM-bound floor, the comparatively flat GPU+copy path wins instead.
// On this machine that crossover measured out to roughly 150-256 MiB -- this
// is hardware- and buffer-size-specific, so re-run this benchmark at your
// actual production sizes rather than assuming the same threshold elsewhere.
//
// The same L3 spillover pattern was measured on a 2-socket Intel Xeon Gold
// 6338T server (24 cores/socket, 1.25 MiB L2/core, 36 MiB L3/socket) with an
// NVIDIA L4 on PCIe 4.0 x16. The crossover was roughly 36-40 MiB; at 4096x4096
// floats (64 MiB), host memset took ~4.04 ms versus ~2.75 ms for GPU+copy.
// ---------------------------------------------------------------------------
#include "containers/image.cuh"
#include "containers/streams.cuh"
#include "timer.h"

#include <algorithm>
#include <cstring>
#include <limits>

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

  HighResolutionTimer<std::chrono::microseconds> timer;

  // Warm up both paths once so we don't measure one-off setup costs
  // (e.g. first-touch page faults, driver/context warmup).
  std::memset(h_img.vec.data().get(), 0, numBytes);
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

  return 0;
}

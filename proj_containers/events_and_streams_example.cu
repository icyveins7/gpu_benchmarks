#include <cstdio>
#include <stdexcept>

#include "containers/events.cuh"
#include "containers/image.cuh"
#include "containers/streams.cuh"

__global__ void fillKernel(containers::Image<double> img, double value) {
  for (int y = threadIdx.y; y < img.height; y += blockDim.y) {
    for (int x = threadIdx.x; x < img.width; x += blockDim.x) {
      img.at(y, x) = value;
    }
  }
}

int main() {
  printf("Testing events and streams\n");

  int width = 1024, height = 1024;
  containers::PinnedHostImageStorage<double> h_img(width, height);
  containers::DeviceImageStorage<double> d_img(width, height);

  containers::CudaStream computeStream;
  containers::CudaStream memcpyStream;
  containers::CudaEvent event;

  // Run a simple kernel with just 1 block (to be slow)
  fillKernel<<<1, dim3(32, 32), 0, computeStream()>>>(d_img.image(), 3.14);

  // Record event
  event.record(computeStream());

  // Try to re-record immediately — kernel is likely still running, so this
  // should throw
  try {
    event.recordWithoutOverwrite(computeStream());
    printf("recordWithoutOverwrite: did NOT throw (event already finished)\n");
  } catch (const std::runtime_error &e) {
    printf("recordWithoutOverwrite: caught expected throw: %s\n", e.what());
  }

  // Wait on event and copy down
  memcpyStream.wait(event());
  d_img.toHost(h_img, memcpyStream());

  // Sync the memcpy stream so we can read the host data
  memcpyStream.sync();

  // Verify
  bool pass = true;
  for (int i = 0; i < width * height; ++i) {
    if (h_img.vec[i] != 3.14) {
      printf("FAIL at index %d: expected 3.14, got %f\n", i, h_img.vec[i]);
      pass = false;
      break;
    }
  }
  printf("%s\n", pass ? "PASS" : "FAIL");

  // Event should report success after sync
  printf("Event query: %s\n",
         event.query() == cudaSuccess ? "cudaSuccess" : "cudaErrorNotReady");

  return 0;
}

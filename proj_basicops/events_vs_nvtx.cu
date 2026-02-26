#include <nvtx3/nvToolsExt.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

int main(int argc, char *argv[]) {
  size_t len = 10;
  if (argc > 1) {
    len = std::atoi(argv[1]);
  }
  printf("Using length %zu\n", len);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  thrust::device_vector<int> d_x(len);

  // warmup
  thrust::sequence(d_x.begin(), d_x.end());
  thrust::fill(d_x.begin(), d_x.end(), 0);

  nvtxRangePush("operations");
  cudaEventRecord(start);
  thrust::sequence(d_x.begin(), d_x.end());
  thrust::fill(d_x.begin(), d_x.end(), 0);
  cudaEventRecord(stop);
  nvtxRangePop();

  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Time: %f ms\n", ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}

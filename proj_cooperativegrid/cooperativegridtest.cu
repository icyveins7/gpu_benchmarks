#include "gridsync.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>

extern "C" {
__global__ void simple_kernel(const int *x, const int *y, int *z,
                              const int length) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < length;
       i += blockDim.x * gridDim.x) {
    z[i] = x[i] + y[i];
  }
}
}

int main() {
  int device = 0;
  bool coopLaunchSupported = checkCooperativeLaunchSupported(device);
  if (coopLaunchSupported)
    std::cout << "Cooperative launch supported for device " << device
              << std::endl;
  else {
    std::cout << "Cooperative launch not supported for device " << device
              << std::endl;

    return -1;
  }

  int length = 10000000;
  int threads = 128;

  thrust::device_vector<int> x(length);
  thrust::device_vector<int> y(length);
  thrust::device_vector<int> z(length);

  int blksPerSm = getMaxBlocksPerSmForCooperativeGrid(threads, simple_kernel);
  printf("Blks per SM = %d\n", blksPerSm);

  int blks = getMaxBlocksForCooperativeGrid(threads, simple_kernel);
  printf("Blks for whole grid = %d\n", blks);

  simple_kernel<<<blks, threads>>>(thrust::raw_pointer_cast(x.data()),
                                   thrust::raw_pointer_cast(y.data()),
                                   thrust::raw_pointer_cast(z.data()), length);

  return 0;
}

/*
This example shows that the mempool's maxSize is not respected. Online searches
also indicate that the maxSize is simply a suggestion, and that the pool maxSize
is not a hard ceiling.
*/

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "containers/streams.cuh"

__global__ void dummyKernel(uint8_t *buf, int count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    buf[i] += 1;
  }
}

int main(int argc, char **argv) {
  printf("Limited cuda mempool experiment\n");
  cudaError_t err;

  cudaMemPool_t pool;
  cudaMemPoolProps poolProps;
  poolProps.allocType = cudaMemAllocationTypePinned;
  poolProps.handleTypes = cudaMemHandleTypeNone;
  poolProps.location.type = cudaMemLocationTypeDevice;
  poolProps.location.id = 0; // device id 0
  poolProps.maxSize = 1000;

  if ((err = cudaMemPoolCreate(&pool, &poolProps)) != cudaSuccess) {
    throw std::runtime_error("cudaMemPoolCreate failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  containers::CudaStream stream;

  uint8_t *d_buf1;
  uint8_t *d_buf2;

  if (cudaMallocAsync(&d_buf1, 1000, pool, stream()) != cudaSuccess) {
    throw std::runtime_error("cudaMallocAsync buf1 failed");
  } else {
    printf("buf1 allocated successfully\n");
  }

  if (cudaMallocAsync(&d_buf2, 1000, pool, stream()) != cudaSuccess) {
    throw std::runtime_error("cudaMallocAsync buf2 failed");
  } else {
    printf("buf2 allocated successfully\n");
  }

  dummyKernel<<<1, 1, 0, stream()>>>(d_buf1, 1000);
  dummyKernel<<<1, 1, 0, stream()>>>(d_buf2, 1000);

  cudaFreeAsync(d_buf1, stream());
  cudaFreeAsync(d_buf2, stream());
  cudaMemPoolDestroy(pool);
  return 0;
}

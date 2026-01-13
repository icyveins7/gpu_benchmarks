/*
This example shows that the mempool's maxSize is not respected. Online searches
also indicate that the maxSize is simply a suggestion, and that the pool maxSize
is not a hard ceiling.

What it does seem to do is to limit what a *single* allocation maximum size can
be i.e. a single contiguous segment cannot be longer than maxSize, otherwise it
will fail.

The below example now shows that *assuming we order the mallocs and frees to the
pool in a serial manner*, submissions to the stream are completely serialized as
expected and occur almost instantaneously (visible when profiled).
*/

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <thrust/host_vector.h>

#include "containers/streams.cuh"
#include "pinnedalloc.cuh"

__global__ void dummyKernel(uint8_t *buf, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
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
  poolProps.maxSize = 15000000000;

  printf("Allocating cudaMemPool...");
  if ((err = cudaMemPoolCreate(&pool, &poolProps)) != cudaSuccess) {
    throw std::runtime_error("cudaMemPoolCreate failed: " +
                             std::string(cudaGetErrorString(err)));
  } else {
    printf("Done\n");
  }

  containers::CudaStream stream;

  uint8_t *d_buf1;
  uint8_t *d_buf2;
  size_t bufSize = 15000000000;
  printf("Allocating pinned host vectors..");
  thrust::pinned_host_vector<uint8_t> h_buf1(bufSize);
  thrust::pinned_host_vector<uint8_t> h_buf2(bufSize);
  printf("Done.\n");

  printf("cudaMallocAsync for buf1..\n");
  if ((err = cudaMallocAsync(&d_buf1, bufSize, pool, stream())) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMallocAsync buf1 failed: " +
                             std::string(cudaGetErrorString(err)));
  } else {
    printf("buf1 allocated successfully\n");
  }
  dummyKernel<<<bufSize / 1024, 1024, 0, stream()>>>(d_buf1, bufSize);
  cudaMemcpyAsync(h_buf1.data().get(), d_buf1, bufSize, cudaMemcpyDeviceToHost,
                  stream());
  printf("CUDA Error for kernel? %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFreeAsync(d_buf1, stream());

  printf("cudaMallocAsync for buf2..\n");
  if ((err = cudaMallocAsync(&d_buf2, bufSize, pool, stream())) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMallocAsync buf2 failed: " +
                             std::string(cudaGetErrorString(err)));
  } else {
    printf("buf2 allocated successfully\n");
  }
  dummyKernel<<<bufSize / 1024, 1024, 0, stream()>>>(d_buf2, bufSize);
  cudaMemcpyAsync(h_buf2.data().get(), d_buf2, bufSize, cudaMemcpyDeviceToHost,
                  stream());
  printf("CUDA Error for kernel? %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFreeAsync(d_buf2, stream());

  // Explicitly synchronize?
  stream.sync();

  cudaMemPoolDestroy(pool);
  return 0;
}

// clang-format off
// nvcc -std=c++17 -I../include -o shared_memory_violations ../proj_slipups/shared_memory_violations.cu

#include "sharedmem.cuh"
#include <thrust/device_vector.h>

template <typename T> __global__ void kernel1(const T *a, T *b) {
  SharedMemory<T> smem;
  T *ptr = smem.getPointer();
  ptr[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
  printf("ptr[%d] = %d\n", threadIdx.x, ptr[threadIdx.x]);
  __syncthreads();

  b[blockIdx.x * blockDim.x + threadIdx.x] =
      ptr[(threadIdx.x + 1) % blockDim.x];
}

int main() {
  dim3 tpb(32);
  dim3 bpg(1);

  thrust::device_vector<int> d_a(32);
  thrust::device_vector<int> d_b(32);

  kernel1<<<bpg, tpb>>>(thrust::raw_pointer_cast(d_a.data()),
                        thrust::raw_pointer_cast(d_b.data()));

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  return 0;
}

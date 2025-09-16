// clang-format off
// nvcc -std=c++17 -I../include -o shared_memory_violations ../proj_slipups/shared_memory_violations.cu
// compute-sanitizer emits no errors??

#include "sharedmem.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

template <typename T> __global__ void kernel1(const T *a, T *b) {
  SharedMemory<T> smem;
  T *ptr = smem.getPointer();
  ptr[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
  printf("ptr[%d] = %d\n", threadIdx.x, ptr[threadIdx.x]);
  __syncthreads();

  // Some non-trivial work to make sure not everything is optimized away?
  b[blockIdx.x * blockDim.x + threadIdx.x] =
      ptr[(threadIdx.x + 1) % blockDim.x];
}

__global__ void fixedkernel(const int *a, int *b){
  extern __shared__ int smem[];
  int *ptr = &smem[0];
  ptr[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
  printf("fixedkernel: ptr[%d], %p = %d\n", threadIdx.x, &ptr[threadIdx.x], ptr[threadIdx.x]);
  __syncthreads();

  // Some non-trivial work to make sure not everything is optimized away?
  b[blockIdx.x * blockDim.x + threadIdx.x] =
      ptr[(threadIdx.x + 1) % blockDim.x];

}

int main() {
  dim3 tpb(32);
  dim3 bpg(1);

  thrust::device_vector<int> d_a(32);
  thrust::sequence(d_a.begin(), d_a.end());
  thrust::device_vector<int> d_b(32);

  // Don't dynamically allocate any shared mem
  // kernel1<<<bpg, tpb>>>(thrust::raw_pointer_cast(d_a.data()),
  //                       thrust::raw_pointer_cast(d_b.data()));
  fixedkernel<<<bpg, tpb>>>(thrust::raw_pointer_cast(d_a.data()),
                            thrust::raw_pointer_cast(d_b.data()));

  cudaDeviceSynchronize();


  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
  }

  thrust::host_vector<int> h_b = d_b;
  for (int i = 0; i < 32; ++i) {
    printf("d_b[%d] = %d\n", i, h_b[i]);
  }

  return 0;
}

#include <iostream>
#include <random>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
namespace cg = cooperative_groups;

template <typename T, int THREADS_PER_BLK>
__global__ void cubWarpReduceKernel(const T *x, const int len, T *out) {
  constexpr int NUM_WARPS = THREADS_PER_BLK / 32; // assume factor of 32

  int blkWarpIdx = threadIdx.x / 32;

  using WarpReduce = cub::WarpReduce<T>;
  __shared__ typename WarpReduce::TempStorage temp_storage[NUM_WARPS];
  for (int i = 0; i < len; i += blockDim.x * gridDim.x) {
    T val = 0;
    int idx = i + threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
      val = x[idx];
    }
    T sum = WarpReduce(temp_storage[blkWarpIdx]).Sum(val);
    if (threadIdx.x % 32 == 0) {
      out[idx / 32] = sum;
    }
  }
}

template <typename T, int THREADS_PER_BLK>
__global__ void cgWarpReduceKernel(const T *x, const int len, T *out) {
  constexpr int NUM_WARPS = THREADS_PER_BLK / 32; // assume factor of 32

  int blkWarpIdx = threadIdx.x / 32;

  auto tilewarp = cg::tiled_partition<32>(cg::this_thread_block());
  for (int i = 0; i < len; i += blockDim.x * gridDim.x) {
    T val = 0;
    int idx = i + threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
      val = x[idx];
    }
    int sum = cg::reduce(tilewarp, val, cg::plus<int>());

    if (tilewarp.thread_rank() == 0) {
      out[idx / 32] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
  printf("CUB vs Cooperative groups\n");

  int len = 1024;
  if (argc >= 2) {
    len = atoi(argv[1]);
  }
  printf("length = %d\n", len);

  thrust::host_vector<int> h_x(len);
  for (auto &x : h_x) {
    x = std::rand() % 10;
  }

  thrust::device_vector<int> d_x = h_x;
  thrust::device_vector<int> d_y(d_x.size());

  {
    constexpr int tpb = 128;
    int numBlks = len / tpb + 1;
    cubWarpReduceKernel<int, tpb>
        <<<numBlks, tpb>>>(d_x.data().get(), len, d_y.data().get());
    printf("Warp cub storage = %zu\n",
           sizeof(typename cub::WarpReduce<int>::TempStorage));
  }

  thrust::host_vector<int> h_y = d_y;
  for (int i = 0; i < len; i += 32) {
    int check = 0;
    for (int j = 0; j < 32; j++) {
      if (i + j < len)
        check += h_x[i + j];
    }
    int oIdx = i / 32;
    if (h_y[oIdx] != check) {
      printf("Error from %d: %d vs %d\n", i, h_y[oIdx], check);
      for (int j = 0; j < 32; j++) {
        printf(" %d, ", h_x[i + j]);
      }
      printf("\n");
      break;
    }
  }

  return 0;
}

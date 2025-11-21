/*
This is the opposite of inflate.cu. Here we look at extracting from a binary map
and getting indices.

Thrust speedup (initially ~10%) is recovered by the kernel by simply halving the
number of blocks i.e. grid stride but with half coverage, probably so each
thread can hide some latency.

For large numbers of active indices, the struct comparison favours the kernel
heavily. The transforms required to wrangle the thrust::gather to work with the
struct are probably not optimal.
*/

#include <algorithm>
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

template <typename T>
void printResult(thrust::host_vector<T> &h_in, thrust::host_vector<T> &h_out,
                 thrust::host_vector<int> &h_activeIdx) {
  printf("--- Input list ---\n");
  for (size_t i = 0; i < h_in.size(); ++i) {
    if (h_in[i] >= 0)
      std::cout << h_in[i] << " ";
    else
      std::cout << "-" << " ";
  }
  printf("\n");

  printf("--- Active indices ---\n");
  for (size_t i = 0; i < h_activeIdx.size(); ++i) {
    printf("%d ", h_activeIdx[i]);
  }
  printf("\n");

  printf("--- Output --- \n");
  for (size_t i = 0; i < h_out.size(); ++i) {
    std::cout << h_out[i] << " ";
  }
  printf("\n");

  printf("=================\n");
}

template <typename T> struct CountingToStrideFunctor {
  T stride;
  __host__ __device__ T operator()(const T &i) const { return i * stride; }
};

template <typename T>
__global__ void deflate_kernel(const T *d_in, const int *idx, const int numIdx,
                               T *out) {
  // Assume out has sufficient length i.e. > numIdx
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numIdx;
       i += blockDim.x * gridDim.x) {
    out[i] = d_in[idx[i]];
  }
}

template <typename T> struct Container {
  int idx;
  T val;

  __host__ __device__ int operator()(const Container &c) const { return c.idx; }
};

template <typename T>
__global__ void deflate_to_struct_kernel(const T *d_in,
                                         Container<T> *containers,
                                         const int numContainers) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numContainers;
       i += blockDim.x * gridDim.x) {
    containers[i].val = d_in[containers[i].idx];
  }
}

template <typename T>
void test(const size_t totalSize, const size_t activeNumber) {
  // make vectors
  thrust::host_vector<T> h_in(totalSize);
  thrust::fill(h_in.begin(), h_in.end(), -1);
  thrust::minstd_rand rng;
  thrust::uniform_int_distribution<T> dist(1, 9);

  // make first activeNumber random numbers
  for (size_t i = 0; i < activeNumber; ++i) {
    h_in[i] = dist(rng);
  }
  // shuffle so these are randomly placed
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(h_in.begin(), h_in.end(), g);
  // to device
  thrust::device_vector<T> d_in(h_in.size());
  thrust::copy(h_in.begin(), h_in.end(), d_in.begin());

  // create the active indices answer
  thrust::host_vector<int> h_activeIdx(activeNumber);
  size_t ctr = 0;
  for (size_t i = 0; i < h_in.size(); ++i) {
    if (h_in[i] >= 0)
      h_activeIdx[ctr++] = i;
  }

  // to device
  thrust::device_vector<int> d_activeIdx(h_activeIdx.size());
  thrust::copy(h_activeIdx.begin(), h_activeIdx.end(), d_activeIdx.begin());

  // make output with number of active indices
  thrust::device_vector<T> d_out(h_activeIdx.size());
  thrust::host_vector<T> h_out(h_activeIdx.size());

  // 1a. call gather
  thrust::fill(d_out.begin(), d_out.end(), 0); // reset
  thrust::gather(d_activeIdx.begin(), d_activeIdx.end(), d_in.begin(),
                 d_out.begin());
  h_out = d_out;

  if (h_in.size() < 1000 && h_activeIdx.size() < 100) {
    printResult(h_in, h_out, h_activeIdx);
  }

  // 1b. use kernel
  thrust::fill(d_out.begin(), d_out.end(), 0); // reset
  int blks = h_activeIdx.size() / 128 + (h_activeIdx.size() % 128 > 0 ? 1 : 0);
  blks /= 2; // follow thrust and use a half-grid
  deflate_kernel<<<blks, 128>>>(d_in.data().get(), d_activeIdx.data().get(),
                                d_activeIdx.size(), d_out.data().get());

  h_out = d_out;
  if (h_in.size() < 1000 && h_activeIdx.size() < 100) {
    printResult(h_in, h_out, h_activeIdx);
  }

  // 2a. call gather but with writes to a container
  thrust::host_vector<Container<T>> h_containers(h_activeIdx.size());
  for (size_t i = 0; i < h_containers.size(); ++i) {
    h_containers[i].idx = h_activeIdx[i];
    h_containers[i].val = std::numeric_limits<T>::max();
  }
  thrust::device_vector<Container<T>> d_containers = h_containers;

  // create an offsetted pointer to the value field
  T *val_ptr = reinterpret_cast<T *>(
      reinterpret_cast<char *>(thrust::raw_pointer_cast(&d_containers[0])) +
      offsetof(Container<T>, val));
  // compute the stride based on the struct size
  int ptr_stride = (int)sizeof(Container<T>);
  printf("ptr_stride = %d\n", ptr_stride);
  // create a device pointer to use for iterating
  thrust::device_ptr<T> d_container_value_ptr(val_ptr);
  // and create a strided iterator for it
  // NOTE: strided_iterator is very new (mid 2025) so we can try to make our own
  // from zip, perm and transforms
  // NOTE: although this gives correct results, i would argue this makes it very
  // unclear as compared to the kernel method
  auto strideFunctor =
      CountingToStrideFunctor<int>{ptr_stride / (int)sizeof(T)};
  auto perm_indices_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0), strideFunctor);
  auto perm_iter = thrust::make_permutation_iterator(d_container_value_ptr,
                                                     perm_indices_iter);

  thrust::gather(
      // On reading we transform by reading just the index from the struct
      thrust::make_transform_iterator(d_containers.begin(), Container<T>()),
      thrust::make_transform_iterator(d_containers.end(), Container<T>()),
      d_in.begin(),
      perm_iter // use custom permutation iterator to stride access to just
                // values
  );

  h_containers = d_containers;
  // copy all the values to h_out for convenience
  for (size_t i = 0; i < h_containers.size(); ++i) {
    h_out[i] = h_containers[i].val;
  }

  if (h_in.size() < 1000 && h_activeIdx.size() < 100) {
    printResult(h_in, h_out, h_activeIdx);
  }

  // 2b. use kernel to write to structs
  for (size_t i = 0; i < h_containers.size(); ++i) {
    h_containers[i].idx = h_activeIdx[i];
    h_containers[i].val = std::numeric_limits<T>::max();
  } // reset
  d_containers = h_containers;

  deflate_to_struct_kernel<<<blks, 128>>>(
      d_in.data().get(), d_containers.data().get(), d_containers.size());

  h_containers = d_containers;
  for (size_t i = 0; i < h_containers.size(); ++i) {
    h_out[i] = h_containers[i].val;
  }

  if (h_in.size() < 1000 && h_activeIdx.size() < 100) {
    printResult(h_in, h_out, h_activeIdx);
  }
}

int main(int argc, char **argv) {
  printf("Testing deflation from lists.\n");
  if (argc != 3) {
    printf("Usage: inflate <totalSize> <activeNumber>\n");
    return 1;
  }
  size_t totalSize = atoi(argv[1]);
  size_t activeNumber = atoi(argv[2]);

  test<int>(totalSize, activeNumber);

  return 0;
}

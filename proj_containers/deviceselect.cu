#include "containers/cubwrappers.cuh"
#include "sharedmem.cuh"

#include <cuda_runtime.h>
#include <iostream>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

// Compile-time toggle to use giant struct for tests
#define USE_BIG_DATA

template <typename T> struct LessThan {
  T value;

  __host__ __device__ bool operator()(const T &a) const { return (a < value); }
};

// NOTE: using large struct like this does affect the workspace
// mem requirements, but it only starts at a certain point (for small numbers
// like length 64, it is no different to applying to raw ints for example)
template <typename T> struct BigData {
  T val;
  T a1;
  T a2;
  T a3;
  T a4;
  T a5;
  T a6;
  T a7;
  T a8;
  T a9;
  T a10;
};

template <typename T> struct LessThanBigData {
  T value;

  __host__ __device__ bool operator()(const BigData<T> &a) const {
    return (a.val < value);
  }
};

template <typename T, typename SelectOp>
__global__ void NaiveSelectIf(const T *data, T *output,
                              unsigned int *d_num_selected, int initialLen,
                              const SelectOp select_op) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < initialLen;
       i += blockDim.x * gridDim.x) {
    if (select_op(data[i])) {
      output[atomicAdd(d_num_selected, 1)] = data[i];
    }
  }
}

template <typename T, typename SelectOp, int ThreadsPerBlk>
__global__ void SelectIfInplaceKernel(T *data, int *d_block_offsets,
                                      unsigned int *d_num_selected,
                                      unsigned int initialLen,
                                      const SelectOp select_op) {

  __shared__ T s_data[ThreadsPerBlk];
  __shared__ unsigned int s_block_count;
  __shared__ unsigned int s_block_offset;
  if (threadIdx.x == 0)
    s_block_count = 0;
  __syncthreads();

  // Read the current block, each thread just does 1
  unsigned int blockReadStart = blockIdx.x * blockDim.x;
  if (blockReadStart >= initialLen) {
    if (threadIdx.x == 0)
      printf("blockIdx.x %d exiting since start = %u\n", blockIdx.x,
             blockReadStart);
    return;
  }
  unsigned int blockReadEnd = blockReadStart + blockDim.x > initialLen
                                  ? initialLen
                                  : blockReadStart + blockDim.x;
  unsigned int numToReadThisBlk = blockReadEnd - blockReadStart;

  // Read data for valid threads
  T threadData;
  if (threadIdx.x < numToReadThisBlk) {
    threadData = data[blockReadStart + threadIdx.x];
    if (select_op(threadData)) {
      s_data[atomicAdd(&s_block_count, 1)] = threadData;
    }
  }

  if (threadIdx.x == 0) {
    s_block_offset = 0; // default is 0 i.e. block 0
    if (blockIdx.x > 0) {
      // Spin wait on previous block
      while (__ldcg(&d_block_offsets[blockIdx.x - 1]) < 0) {
        __threadfence();
      }
      // Update
      s_block_offset = d_block_offsets[blockIdx.x - 1];
    }
  }
  __syncthreads(); // need all threads to see the block offset

  // Write data for each block
  if (threadIdx.x < s_block_count) {
    data[s_block_offset + threadIdx.x] = s_data[threadIdx.x];
  }

  // Update the next block
  if (threadIdx.x == 0) {
    __stcs(&d_block_offsets[blockIdx.x], s_block_offset + s_block_count);
    // and the total count
    atomicAdd(d_num_selected, s_block_count);
  }
}

int main(int argc, char **argv) {
  printf("One block device select comparisons\n");

#ifdef USE_BIG_DATA
  using DataT = BigData<int>;
  printf("Using big data struct\n");
#else
  using DataT = int;
#endif
  using NumSelectedT = unsigned int;

  size_t length = 8192;
  if (argc > 1) {
    length = atoi(argv[1]);
  }
  printf("length: %zu\n", length);
  thrust::host_vector<DataT> h_data(length);
  thrust::host_vector<NumSelectedT> h_num_selected(1);
  for (int i = 0; i < (int)length; i++) {
#ifdef USE_BIG_DATA
    // Fill just first value in the struct
    h_data[i].val = i % 10;
#else
    // Fill ints
    h_data[i] = i % 10;
#endif
  }

  thrust::device_vector<DataT> d_data = h_data;
  thrust::device_vector<DataT> d_out(d_data.size());
  thrust::device_vector<NumSelectedT> d_num_selected(1);

#ifdef USE_BIG_DATA
  LessThanBigData<int> functor{5};
#else
  LessThan<DataT> functor{5};
#endif

  {
    cubw::DeviceSelect::IfInPlace<decltype(d_data.begin()), NumSelectedT *,
                                  decltype(functor)>
        if_inplace(length);
    printf("Temp storage is %zu bytes\n", if_inplace.d_temp_storage.size());
    if_inplace.exec(d_data.begin(), d_num_selected.data().get(), length,
                    functor);
    for (int i = 0; i < 3; ++i) {
      d_data = h_data;
      if_inplace.exec(d_data.begin(), d_num_selected.data().get(), length,
                      functor);
    }
  }
  h_num_selected = d_num_selected;
  printf("Num selected: %d\n", h_num_selected[0]);
  thrust::host_vector<DataT> h_out = d_data;
  for (int i = 0; i < (int)h_num_selected[0]; i++) {
#ifdef USE_BIG_DATA
    if (h_out[i].val >= functor.value) {
#else
    if (h_out[i] >= functor.value) {
#endif
      std::cout << "Error at index " << i << std::endl;
    }
    if (length <= 64) {
#ifdef USE_BIG_DATA
      std::cout << h_out[i].val << " ";
#else
      std::cout << h_out[i] << " ";
#endif
    }
  }
  std::cout << std::endl << " -------------- " << std::endl;

  {
    d_data = h_data;
    cubw::DeviceSelect::If<decltype(d_data.begin()), decltype(d_out.begin()),
                           NumSelectedT *, decltype(functor)>
        if_outofplace(length);
    printf("Temp storage is %zu bytes\n", if_outofplace.d_temp_storage.size());
    for (int i = 0; i < 3; ++i) {
      if_outofplace.exec(d_data.begin(), d_out.begin(),
                         d_num_selected.data().get(), length, functor);
    }
  }
  h_num_selected = d_num_selected;
  printf("Num selected: %d\n", h_num_selected[0]);
  h_out = d_out;
  for (int i = 0; i < (int)h_num_selected[0]; i++) {
#ifdef USE_BIG_DATA
    if (h_out[i].val >= functor.value) {
#else
    if (h_out[i] >= functor.value) {
#endif
      std::cout << "Error at index " << i << std::endl;
    }
    if (length <= 64) {
#ifdef USE_BIG_DATA
      std::cout << h_out[i].val << " ";
#else
      std::cout << h_out[i] << " ";
#endif
    }
  }
  std::cout << std::endl << "-----------------" << std::endl;

  // Attempt custom naive out of place kernel
  constexpr int tpb = 128;
  int blks = (length + tpb - 1) / tpb;
  d_num_selected[0] = 0;
  NaiveSelectIf<<<blks, tpb>>>(d_data.data().get(), d_out.data().get(),
                               d_num_selected.data().get(), length, functor);
  h_num_selected = d_num_selected;
  printf("Num selected: %d\n", h_num_selected[0]);
  h_out = d_out;
  for (int i = 0; i < (int)h_num_selected[0]; i++) {
#ifdef USE_BIG_DATA
    if (h_out[i].val >= functor.value) {
#else
    if (h_out[i] >= functor.value) {
#endif
      std::cout << "Error at index " << i << std::endl;
    }
    if (length <= 64) {
#ifdef USE_BIG_DATA
      std::cout << h_out[i].val << " ";
#else
      std::cout << h_out[i] << " ";
#endif
    }
  }
  std::cout << std::endl << "-----------------" << std::endl;

  // Attempt in place kernel
  for (int i = 0; i < 3; ++i) {
    d_num_selected[0] = 0;
    d_data = h_data;
    thrust::device_vector<int> d_block_offsets(blks, -1);
    SelectIfInplaceKernel<DataT, decltype(functor), tpb>
        <<<blks, tpb>>>(d_data.data().get(), d_block_offsets.data().get(),
                        d_num_selected.data().get(), length, functor);
  }
  h_num_selected = d_num_selected;
  printf("Num selected: %d\n", h_num_selected[0]);
  h_out = d_data;
  for (int i = 0; i < (int)h_num_selected[0]; i++) {
#ifdef USE_BIG_DATA
    if (h_out[i].val >= functor.value) {
#else
    if (h_out[i] >= functor.value) {
#endif
      std::cout << "Error at index " << i << std::endl;
    }
    if (length <= 64) {
#ifdef USE_BIG_DATA
      std::cout << h_out[i].val << " ";
#else
      std::cout << h_out[i] << " ";
#endif
    }
  }
  std::cout << std::endl << "-----------------" << std::endl;

  return 0;
}

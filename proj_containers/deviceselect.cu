#include "containers/cubwrappers.cuh"
#include "sharedmem.cuh"
#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <iostream>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

namespace cg = cooperative_groups;

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
__global__ void SelectIfInplaceKernel(T *data, volatile int *d_block_offsets,
                                      volatile int *d_block_prefixes,
                                      unsigned int *d_num_selected,
                                      unsigned int initialLen,
                                      const SelectOp select_op) {

  __shared__ T s_data[ThreadsPerBlk];
  __shared__ unsigned int s_block_count;
  __shared__ unsigned int s_block_prefix;
  if (threadIdx.x == 0) {
    s_block_count = 0;
    s_block_prefix = 0;
  }
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
  int sIdx = -1;
  if (threadIdx.x < numToReadThisBlk) {
    threadData = data[blockReadStart + threadIdx.x];
    if (select_op(threadData)) {
      sIdx = atomicAdd(&s_block_count, 1);
      // printf("thread %d sIdx %d\n", threadIdx.x, sIdx);
      // NOTE: we haven't written to shared mem yet
      // we just want to publish our prefixes and offsets asap so we delay this
    }
  }
  __syncthreads();

  // Thread 0: Populate the block offset, but not the prefix yet
  // We arrange the offsetprefixes in reverse i.e. block 0 writes to the last
  // element This makes it easier to 'look back' as it would be equivalent to
  // incrementing the index
  unsigned int thisBlockOffsetPrefixesIndex = gridDim.x - 1 - blockIdx.x;
  if (threadIdx.x == 0) {
    // We have the inclusive sum for block 0 already
    if (blockIdx.x == 0) {
      d_block_prefixes[thisBlockOffsetPrefixesIndex] = (int)s_block_count;
    }
    __threadfence(); // prefix should be written first
    d_block_offsets[thisBlockOffsetPrefixesIndex] = (int)s_block_count;
    // Programming guide suggests just marking it volatile means i don't need a
    // special builtin like __stcs:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions
  }

  // Now we fill the shared mem
  if (sIdx >= 0) {
    s_data[sIdx] = threadData;
  }

  // First thread: decoupled lookback
  int rollingPrefix = 0;
  if ((threadIdx.x == 0) && (blockIdx.x != 0)) {
    bool complete = false;
    int blkIdx = thisBlockOffsetPrefixesIndex + 1;
    while (!complete) {
      int prevBlkPrefix = d_block_prefixes[blkIdx];
      if (prevBlkPrefix >= 0) {
        // Block has completed its own prefix
        complete = true;
        rollingPrefix += prevBlkPrefix;
        break;
      }

      int prevBlkOffset = d_block_offsets[blkIdx];
      if (prevBlkOffset < 0) {
        // Block has not completed its own offset yet
        __threadfence();
        // We repeat on the same block
        continue;
      }
      // Otherwise we increment
      rollingPrefix += prevBlkOffset;

      // Otherwise we look back further
      if (blkIdx < (int)gridDim.x - 1)
        blkIdx++;
    }
    s_block_prefix = rollingPrefix;
    // printf("Block %d prefix should be %d\n", blockIdx.x, s_block_prefix);
    // Update the block prefixes for next blocks
    d_block_prefixes[thisBlockOffsetPrefixesIndex] =
        rollingPrefix + s_block_count;
  }
  __syncthreads(); // wait for all threads to know where to write

  // // First warp: decoupled lookback
  // if (threadIdx.x < 32) {
  //   if (blockIdx.x > 0) {
  //     bool complete = false;
  //     while (!complete) {
  //       // Lookback until we hit something with no block offset
  //       // NOTE: now to look back, we just go forwards
  //       for (int i = threadIdx.x;
  //            i < (int)gridDim.x - (int)thisBlockOffsetPrefixesIndex - 1;
  //            i += 32) {
  //         // First thread on first loop should read the +1 block, and so on
  //         int prevBlkOffset =
  //             (int)d_block_offsets[i + 1 + thisBlockOffsetPrefixesIndex];
  //         bool prevBlkOffsetWritten = (prevBlkOffset >= 0);
  //         bool allOffsetsFound = false;
  //         if (prevBlkOffsetWritten) {
  //           cg::coalesced_group g = cg::coalesced_threads();
  //           g.size()
  //         }
  //         // __threadfence(); // maybe i dont need this any more?
  //       }
  //     }
  //   }
  // }

  // Write data for each block
  if (threadIdx.x < s_block_count) {
    data[s_block_prefix + threadIdx.x] = s_data[threadIdx.x];
  }

  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
    // Update the global sum
    atomicAdd(d_num_selected, s_block_prefix + s_block_count);
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
  constexpr int tpb = 128; // doesn't make much diff for out of place, but makes
                           // in-place kernel below much worse if you reduce
  int blks = length / tpb + (length % tpb > 0 ? 1 : 0);
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
    thrust::device_vector<int> d_block_prefixes(blks, -1);
    SelectIfInplaceKernel<DataT, decltype(functor), tpb>
        <<<blks, tpb>>>(d_data.data().get(), d_block_offsets.data().get(),
                        d_block_prefixes.data().get(),
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
      std::cout << "Stopping here..." << std::endl;
      break;
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

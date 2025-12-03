#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/limits>

#include <cub/cub.cuh>

#include "containers/cubwrappers.cuh"

#include <nvtx3/nvToolsExt.h>

template <typename T, int NUM_THREADS, int ITEMS_PER_THREAD>
__global__ void OneBlockSortKeys(const int *d_keys_in, int *d_keys_out,
                                 const int num_items) {
  using BlockRadixSort = cub::BlockRadixSort<T, NUM_THREADS, ITEMS_PER_THREAD>;
  using TempStorageSort = typename BlockRadixSort::TempStorage;

  using BlockExchange = cub::BlockExchange<T, NUM_THREADS, ITEMS_PER_THREAD>;
  using TempStorageExchange = typename BlockExchange::TempStorage;

  union ShmemLayout {
    TempStorageSort sort;
    TempStorageExchange exchange;
  };
  // Just copied from the example
  extern __shared__ __align__(alignof(ShmemLayout)) char shmem[];

  // Cast to the appropriate type
  auto &temp_storage = reinterpret_cast<TempStorageSort &>(shmem);

  int thread_keys[ITEMS_PER_THREAD];
  for (int t = threadIdx.x; t < NUM_THREADS * ITEMS_PER_THREAD;
       t += NUM_THREADS) {
    if (t < num_items) {
      thread_keys[t / NUM_THREADS] = d_keys_in[t];
    } else {
      thread_keys[t / NUM_THREADS] = cuda::std::numeric_limits<T>::max();
    }
  }
  BlockRadixSort(temp_storage).Sort(thread_keys);
  __syncthreads();

  // Exchange to striped before writing back
  // Cast to the appropriate type
  auto &temp_storage2 = reinterpret_cast<TempStorageExchange &>(shmem);
  BlockExchange(temp_storage2).BlockedToStriped(thread_keys);

  for (int t = threadIdx.x; t < num_items; t += NUM_THREADS) {
    d_keys_out[t] = thread_keys[t / NUM_THREADS];
  }
}

int main() {
  printf("One block sort comparisons\n");

  using KeyT = int;
  unsigned int num_items;

  for (int i = 5; i <= 20; i++) {
    num_items = 1 << i;
    std::string name = "num_items_" + std::to_string(1 << i);
    nvtxRangePushA(name.c_str());
    printf("num_items: %d\n", num_items);
    thrust::host_vector<KeyT> h_input(num_items);
    for (int i = 0; i < (int)num_items; i++)
      h_input[i] = std::rand() % 1000;
    thrust::device_vector<KeyT> d_input = h_input;
    thrust::device_vector<KeyT> d_output(num_items);

    // Try radix sort
    cubw::DeviceRadixSort::SortKeys<KeyT, int> sort_keys(num_items);

    for (int iter = 0; iter < 3; iter++) {
      sort_keys.exec(d_input.data().get(), d_output.data().get(),
                     (int)num_items);
    }

    thrust::host_vector<int> h_output = d_output;
    for (int i = 1; i < (int)num_items; i++) {
      if (h_output[i - 1] > h_output[i]) {
        std::cout << "radix sort Error at index " << i << std::endl;
      }
    }

    // Try merge sort

    cubw::DeviceMergeSort::SortKeysCopy<KeyT *, KeyT *, int,
                                        cuda::std::less<int>>
        merge_sort_keys(num_items);

    for (int iter = 0; iter < 3; iter++) {
      merge_sort_keys.exec(d_input.data().get(), d_output.data().get(),
                           (int)num_items, cuda::std::less<KeyT>());
    }
    h_output = d_output;
    for (int i = 1; i < (int)num_items; i++) {
      if (h_output[i - 1] > h_output[i]) {
        std::cout << "merge sort Error at index " << i << std::endl;
      }
    }

    // Try the 1 block custom kernel
    const int NUM_THREADS = 256;
    const int ITEMS_PER_THREAD = 32;
    printf("maximum kernel items: %d\n", NUM_THREADS * ITEMS_PER_THREAD);

    // determine necessary storage size
    auto block_sort_temp_bytes =
        sizeof(typename cub::BlockRadixSort<KeyT, NUM_THREADS,
                                            ITEMS_PER_THREAD>::TempStorage);
    auto block_exchg_temp_bytes =
        sizeof(typename cub::BlockExchange<KeyT, NUM_THREADS,
                                           ITEMS_PER_THREAD>::TempStorage);
    printf("block_sort_temp_bytes: %zd\n", block_sort_temp_bytes);
    printf("block_exchg_temp_bytes: %zd\n", block_exchg_temp_bytes);
    auto smem_size = (std::max)(block_sort_temp_bytes, block_exchg_temp_bytes);
    printf("smem_size: %zd\n", smem_size);
    OneBlockSortKeys<KeyT, NUM_THREADS, ITEMS_PER_THREAD>
        <<<1, NUM_THREADS, smem_size>>>(d_input.data().get(),
                                        d_output.data().get(), (int)num_items);

    h_output = d_output;
    if (num_items > NUM_THREADS * ITEMS_PER_THREAD) {
      std::cout << "num_items too large for this kernel" << std::endl;
    } else {
      for (int i = 1; i < (int)num_items; i++) {
        if (h_output[i - 1] > h_output[i]) {
          std::cout << "Custom kernel Error at index " << i << std::endl;
        }
      }
    }
    if (num_items <= 64) {
      for (int i = 1; i < (int)num_items; i++) {
        printf("%d ", h_output[i]);
      }
      printf("\n");
    }
    nvtxRangePop();
    printf("-------------\n");
  }

  return 0;
}

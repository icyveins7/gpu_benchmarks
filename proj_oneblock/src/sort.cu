#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/limits>

#include <cub/cub.cuh>

#include "containers/cubwrappers.cuh"

#include <nvtx3/nvToolsExt.h>

struct ValueContainer64 {
  unsigned long long int a1;
  unsigned long long int a2;
  unsigned long long int a3;
  unsigned long long int a4;
  unsigned long long int a5;
  unsigned long long int a6;
  unsigned long long int a7;
  unsigned long long int a8;

  void setAll(unsigned long long int val) {
    a1 = val;
    a2 = val;
    a3 = val;
    a4 = val;
    a5 = val;
    a6 = val;
    a7 = val;
    a8 = val;
  }
};

template <typename T, int NUM_THREADS, int ITEMS_PER_THREAD>
__global__ void OneBlockSortKeys(const T *d_keys_in, T *d_keys_out,
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

  T thread_keys[ITEMS_PER_THREAD];
  for (int t = threadIdx.x; t < NUM_THREADS * ITEMS_PER_THREAD;
       t += NUM_THREADS) {
    if (t < num_items) {
      thread_keys[t / NUM_THREADS] = d_keys_in[t];
    } else {
      thread_keys[t / NUM_THREADS] = cuda::std::numeric_limits<T>::max();
    }
  }
  // No need for explicit sync here
  BlockRadixSort(temp_storage).Sort(thread_keys);
  // Sync for smem reuse
  __syncthreads();

  // Exchange to striped before writing back
  // Cast to the appropriate type
  auto &temp_storage2 = reinterpret_cast<TempStorageExchange &>(shmem);
  BlockExchange(temp_storage2).BlockedToStriped(thread_keys);

  for (int t = threadIdx.x; t < min(num_items, NUM_THREADS * ITEMS_PER_THREAD);
       t += NUM_THREADS) {
    d_keys_out[t] = thread_keys[t / NUM_THREADS];
  }
}

template <typename T, int NUM_THREADS, int ITEMS_PER_THREAD, typename V>
__global__ void OneBlockSortPairs(const T *d_keys_in, T *d_keys_out,
                                  const V *d_vals_in, V *d_vals_out,
                                  const int num_items) {
  using IndexType = int;

  using BlockRadixSort =
      cub::BlockRadixSort<T, NUM_THREADS, ITEMS_PER_THREAD, IndexType>;
  using TempStorageSort = typename BlockRadixSort::TempStorage;

  using BlockExchangeKey = cub::BlockExchange<T, NUM_THREADS, ITEMS_PER_THREAD>;
  using TempStorageExchangeKey = typename BlockExchangeKey::TempStorage;

  using BlockExchangeIdx =
      cub::BlockExchange<IndexType, NUM_THREADS, ITEMS_PER_THREAD>;
  using TempStorageExchangeIdx = typename BlockExchangeIdx::TempStorage;

  union ShmemLayout {
    TempStorageSort sort;
    TempStorageExchangeKey exchangeKey;
    TempStorageExchangeIdx exchangeIdx;
  };
  // Just copied from the example
  extern __shared__ __align__(alignof(ShmemLayout)) char shmem[];

  // Cast to the appropriate type
  auto &temp_storage = reinterpret_cast<TempStorageSort &>(shmem);

  T thread_keys[ITEMS_PER_THREAD];
  IndexType thread_idxs[ITEMS_PER_THREAD];

  for (int t = threadIdx.x; t < NUM_THREADS * ITEMS_PER_THREAD;
       t += NUM_THREADS) {
    if (t < num_items) {
      thread_keys[t / NUM_THREADS] = d_keys_in[t];
      thread_idxs[t / NUM_THREADS] = t;
    } else {
      thread_keys[t / NUM_THREADS] = cuda::std::numeric_limits<T>::max();
      thread_idxs[t / NUM_THREADS] = t;
    }
  }
  // No need for explicit sync here
  BlockRadixSort(temp_storage).Sort(thread_keys, thread_idxs);
  // Sync for smem reuse
  __syncthreads();

  // Exchange to striped before writing back
  // Cast to the appropriate type
  auto &temp_storage2 = reinterpret_cast<TempStorageExchangeKey &>(shmem);
  BlockExchangeKey(temp_storage2).BlockedToStriped(thread_keys);
  __syncthreads();

  auto &temp_storage3 = reinterpret_cast<TempStorageExchangeIdx &>(shmem);
  BlockExchangeIdx(temp_storage3).BlockedToStriped(thread_idxs);

  for (int t = threadIdx.x; t < min(num_items, NUM_THREADS * ITEMS_PER_THREAD);
       t += NUM_THREADS) {
    d_keys_out[t] = thread_keys[t / NUM_THREADS];

    int srcIdx = thread_idxs[t / NUM_THREADS];
    if (srcIdx < num_items) {
      d_vals_out[t] = d_vals_in[srcIdx];
    }
  }
}

int main() {
  printf("One block sort comparisons\n");

  using KeyT = int;
  using ValueT = ValueContainer64;
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
    thrust::device_vector<ValueT> d_valinput(num_items);
    thrust::device_vector<ValueT> d_valoutput(num_items);

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

    // Try merge sortpairs

    cubw::DeviceMergeSort::SortPairsCopy<KeyT *, ValueT *, KeyT *, ValueT *,
                                         int, cuda::std::less<KeyT>>
        merge_sort_pairs(num_items);
    for (int iter = 0; iter < 3; iter++) {
      merge_sort_pairs.exec(d_input.data().get(), d_valinput.data().get(),
                            d_output.data().get(), d_valoutput.data().get(),
                            (int)num_items, cuda::std::less<KeyT>());
    }
    h_output = d_output;
    for (int i = 1; i < (int)num_items; i++) {
      if (h_output[i - 1] > h_output[i]) {
        std::cout << "merge sortpairs Error at index " << i << std::endl;
      }
    }

    // Try the 1 block custom kernel
    {
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
      auto smem_size =
          (std::max)(block_sort_temp_bytes, block_exchg_temp_bytes);
      printf("smem_size: %zd\n", smem_size);
      OneBlockSortKeys<KeyT, NUM_THREADS, ITEMS_PER_THREAD>
          <<<1, NUM_THREADS, smem_size>>>(
              d_input.data().get(), d_output.data().get(), (int)num_items);

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
    }

    // Try custom one block sortpairs kernel
    {
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
      auto block_exchg_idx_temp_bytes =
          sizeof(typename cub::BlockExchange<int, NUM_THREADS,
                                             ITEMS_PER_THREAD>::TempStorage);
      printf("block_sort_temp_bytes: %zd\n", block_sort_temp_bytes);
      printf("block_exchg_temp_bytes: %zd\n", block_exchg_temp_bytes);
      printf("block_exchg_idx_temp_bytes: %zd\n", block_exchg_idx_temp_bytes);
      auto smem_size =
          (std::max)(block_sort_temp_bytes, block_exchg_temp_bytes);
      smem_size = (std::max)(smem_size, block_exchg_idx_temp_bytes);
      printf("smem_size: %zd\n", smem_size);
      OneBlockSortPairs<KeyT, NUM_THREADS, ITEMS_PER_THREAD, ValueT>
          <<<1, NUM_THREADS, smem_size>>>(
              d_input.data().get(), d_output.data().get(),
              d_valinput.data().get(), d_valoutput.data().get(),
              (int)num_items);

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
    }

    nvtxRangePop();
    printf("-------------\n");
  }

  return 0;
}

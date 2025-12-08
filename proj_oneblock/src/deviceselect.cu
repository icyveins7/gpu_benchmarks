#include "containers/cubwrappers.cuh"

#include <iostream>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Compile-time toggle to use giant struct for tests
// #define USE_BIG_DATA

template <typename T> struct LessThan {
  T value;

  __host__ __device__ bool operator()(const T &a) const { return (a < value); }
};

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
  thrust::device_vector<NumSelectedT> d_num_selected(1);

#ifdef USE_BIG_DATA
  LessThanBigData<int> functor{5};
#else
  LessThan<DataT> functor{5};
#endif

  cubw::DeviceSelect::IfInPlace<DataT *, NumSelectedT *, decltype(functor)>
      if_inplace(length);
  printf("Temp storage is %zu bytes\n", if_inplace.d_temp_storage.size());
  if_inplace.exec(d_data.data().get(), d_num_selected.data().get(), length,
                  functor);

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
  std::cout << std::endl;

  return 0;
}

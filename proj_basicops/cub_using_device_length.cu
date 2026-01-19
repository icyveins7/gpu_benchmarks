#include <iostream>
#include <limits>
#include <random>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "containers/cubwrappers.cuh"

// attempt to use fancy iterators to work around
// device side length
// #define TEST_1_FANCY_ITERATOR

// use CDP to launch, reading in the device side length
// inside the parent kernel
#define TEST_2_CDP

template <typename T> struct InputWithIndex {
  size_t index;
  T data;
};

template <typename T> struct InputLimitedLengthFunctor {
  T *d_input;
  size_t *d_usedLength;
  T dummyValue;

  __device__ __host__ InputWithIndex<T> operator()(const size_t &index) const {
    if (index < *d_usedLength) {
      return InputWithIndex<T>{index, d_input[index]};
    } else {
      return InputWithIndex<T>{index, dummyValue};
    }
  }
};

template <typename T> struct ComparisonFunctor {
  __device__ __host__ bool operator()(const T &a, const T &b) const {
    auto aval = a.data;
    auto bval = b.data;
    return aval < bval;
  }
};

template <typename Tout, typename Tcomparison>
struct OutputLimitedLengthFunctor {
  Tout *d_output;
  size_t *d_usedLength;

  __device__ __host__ Tcomparison operator()(const Tcomparison &out) const {
    // if (out.index < *d_usedLength) {
    //   d_output[out.index] = out.data; // perform the write directly
    // }
    //
    // return -1; // dummy value to output iterator
    //
    return Tcomparison{out.index + 1, out.data};
  };
};

int main() {
  printf("Testing cub calls using a length kept in a device global memory "
         "scalar.\n");

#if defined(TEST_1_FANCY_ITERATOR)
  size_t length = 100;

  using Tin = int;
  using Tcomparison = InputWithIndex<int>;
  // using Tin = InputWithIndex<int>;
  using Tout = InputWithIndex<int>;

  // Create data
  thrust::host_vector<Tin> h_input(length);
  for (size_t i = 0; i < length; ++i) {
    h_input[i] = std::rand() % 10;
    // h_input[i] = {i, std::rand() % 10};
  }
  thrust::device_vector<Tin> d_input = h_input;

  // Create 'used' length
  thrust::host_vector<size_t> h_usedLength(1);
  h_usedLength[0] = length / 2;
  thrust::device_vector<size_t> d_usedLength = h_usedLength;

  // Allocate output
  thrust::device_vector<Tout> d_output(length);
  thrust::host_vector<Tout> h_output(length);

  // I want to merge sort with the max 'length' but only actually operate on
  // the data kept in 'd_usedLength'.

  // Turn an incrementing index into the indexed data value, or a sentinel value
  // if out of range. Also attach the incrementing index along with it
  auto inputIter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      InputLimitedLengthFunctor<int>{
          d_input.data().get(), d_usedLength.data().get(),
          std::numeric_limits<Tin>::max() // we use a max value since we want it
                                          // to be sorted at the back
      });
  // ==== This is okay

  // Discard the output, but pass the output through the functor which does the
  // length-checked writes
  OutputLimitedLengthFunctor<Tout, Tcomparison> outputFunctor{
      d_output.data().get(), d_usedLength.data().get()};
  auto outputIter = thrust::make_transform_output_iterator(
      d_output.data().get(), outputFunctor);
  // ===== This simply does not work.
  // Looks similar to this issue: https://github.com/NVIDIA/cccl/issues/903

  // invoke the cub calls
  size_t storage_bytes;
  // cub::DeviceMergeSort::SortKeysCopy(nullptr, storage_bytes, inputIter,
  //                                    outputIter, length,
  //                                    ComparisonFunctor<Tcomparison>{});
  // thrust::device_vector<char> d_temp_storage(storage_bytes);
  // cub::DeviceMergeSort::SortKeysCopy(d_temp_storage.data().get(),
  // storage_bytes,
  //                                    inputIter, outputIter, length,
  //                                    ComparisonFunctor<Tcomparison>{});

  // cubw::DeviceMergeSort::SortKeysCopy<decltype(inputIter),
  // decltype(outputIter),
  //                                     size_t, ComparisonFunctor<int>>
  //     cubwSorter(length);
  //
  // cubwSorter.exec(inputIter, outputIter, length, ComparisonFunctor<int>{});

  // Copy out
  h_output = d_output;

  // Check
  for (size_t i = 0; i < length; ++i) {
    printf("%d%c", h_input[i], i == h_usedLength[0] - 1 ? '|' : ' ');
  }
  printf("\n");
  for (size_t i = 0; i < length; ++i) {
    // printf("%d%c", h_output[i], i == h_usedLength[0] - 1 ? '|' : ' ');
    const char *valchar = (h_output[i].data == std::numeric_limits<Tin>::max())
                              ? "X"
                              : std::to_string(h_output[i].data).c_str();
    printf("%s%c", valchar, i == h_usedLength[0] - 1 ? '|' : ' ');
  }
  printf("\n");
#endif

#if defined(TEST_2_CDP)

  using KeyT = unsigned int;
  using NumItemsT = unsigned int;
  NumItemsT length = 100;

  thrust::host_vector<KeyT> h_keys(length);
  for (size_t i = 0; i < length; ++i) {
    h_keys[i] = std::rand() % 10;
  }
  thrust::device_vector<KeyT> d_keys = h_keys;
  thrust::device_vector<KeyT> d_keys_out(d_keys.size(),
                                         std::numeric_limits<KeyT>::max());

  cubw::DeviceRadixSort::SortKeys<KeyT, NumItemsT> cubwSorter(length);

  thrust::host_vector<NumItemsT> h_usedLength = {length / 2};
  thrust::device_vector<NumItemsT> d_usedLength = h_usedLength;
  printf("Launching cdp_exec()\n");
  cubwSorter.cdp_exec(d_keys.data().get(), d_keys_out.data().get(),
                      d_usedLength.data().get());

  cudaDeviceSynchronize();

  thrust::host_vector<KeyT> h_keys_out = d_keys_out;

  // Check
  for (size_t i = 0; i < length; ++i) {
    printf("%u%c", h_keys[i], i == h_usedLength[0] - 1 ? '|' : ' ');
  }
  printf("\n");

  for (size_t i = 0; i < length; ++i) {
    const char *valchar = h_keys_out[i] == std::numeric_limits<KeyT>::max()
                              ? "X"
                              : std::to_string(h_keys_out[i]).c_str();
    printf("%s%c", valchar, i == h_usedLength[0] - 1 ? '|' : ' ');
  }
  printf("\n");

#endif
}

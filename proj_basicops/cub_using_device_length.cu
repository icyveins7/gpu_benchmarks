#include <iostream>
#include <random>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "containers/cubwrappers.cuh"

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
  __device__ __host__ bool operator()(const InputWithIndex<T> &a,
                                      const InputWithIndex<T> &b) const {
    auto aval = a.data;
    auto bval = b.data;
    return aval < bval;
  }
};

template <typename T> struct OutputLimitedLengthFunctor {
  T *d_output;
  size_t *d_usedLength;

  template <typename Tuple> __device__ __host__ T operator()(Tuple out) const {
    int idx = thrust::get<0>(out);
    int val = thrust::get<1>(out);
    if (idx < *d_usedLength) {
      d_output[idx] = val; // perform the write directly
    }

    return -1; // dummy value to output iterator
  };
};

int main() {
  printf("Testing cub calls using a length kept in a device global memory "
         "scalar.\n");

  size_t length = 100;
  thrust::host_vector<int> h_input(length);
  for (size_t i = 0; i < length; ++i) {
    h_input[i] = std::rand() % 10;
  }
  thrust::device_vector<int> d_input = h_input;

  thrust::host_vector<size_t> h_usedLength(1);
  h_usedLength[0] = length / 2;
  thrust::device_vector<size_t> d_usedLength = h_usedLength;

  thrust::device_vector<int> d_output(h_usedLength[0]);
  thrust::host_vector<int> h_output(h_usedLength[0]);

  // I want to merge sort with the max 'length' but only actually operate on
  // the data kept in 'd_usedLength'.

  // Turn an incrementing index into the indexed data value, or a sentinel value
  // if out of range. Also attach the incrementing index along with it
  auto inputIter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      InputLimitedLengthFunctor<int>{d_input.data().get(),
                                     d_usedLength.data().get(), -1});

  // Discard the output, but pass the output through the functor which does the
  // length-checked writes
  // auto outputIter = thrust::make_transform_output_iterator(
  //     thrust::make_discard_iterator(),
  //     OutputLimitedLengthFunctor<int>{d_output.data().get(),
  //                                     d_usedLength.data().get()});
  auto outputIter = thrust::make_discard_iterator();

  // // invoke the cub calls
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
    printf("%d%c", h_output[i], i == h_usedLength[0] - 1 ? '|' : ' ');
  }
  printf("\n");
}

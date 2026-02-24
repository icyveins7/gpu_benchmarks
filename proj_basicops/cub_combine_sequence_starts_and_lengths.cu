/*
This demonstrates that the 'standard' exclusive sum is equivalent to the 'combined' inclusive sum.
In the 'standard' way, we have inputs
1 5 2 3 1
but the exclusive sum inserts a 0, so the logical sequence is actually
0 1 5 2 3|1
The sum is then constructed up to the original length (5), so the output is
0 1 6 8 11

We desire the exclusive sums (starts of each subsequence) and their lengths to be tied together
i.e. at the same index. Hence we want the structs to look like
1 5 2 3 1
0 1 6 8 11

In order to do this we need to first transform the scalar values into their structs, defaulting the start values as 0.
This is done in the input transform iterator, and changes
1 5 2 3 1
to 
1 5 2 3 1
0 0 0 0 0

Note that if we now use an exclusive sum on this struct array, the exclusive sum will insert a 0-like element at the
start again, which would then give
0 1 5 2 3|1
0 0 0 0 0|0

where the last element would be left out of the output. The sums would flow like this:
0 1 5 2 3 1
|\|\|\|\|\
0 0 1 6 8 11

This is the correct order, but due to the exclusive sum adding a blank element we end up with the wrong slice of output.

Instead, what we can do is to simply use an inclusive sum. The summation direction is still the same,
but now we get the slice we want since no additional blank element is added:

1 5 2 3 1
|\|\|\|\
0 1 6 8 11
*/

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <type_traits>

template <typename T> struct SegmentLengthAndStart {
  static_assert(std::is_integral<T>::value, "T must be an integer type");
  T length;
  T start;
  __host__ __device__ SegmentLengthAndStart() : length(0), start(0) {}

  __host__ __device__ SegmentLengthAndStart
  operator+(const SegmentLengthAndStart &other) const {
    SegmentLengthAndStart out;
    out.length = other.length;
    out.start = start + length;
    return out;
  }
};

template <typename T> struct SegmentLengthAndStartTransformer {
  __host__ __device__ SegmentLengthAndStart<T>
  operator()(const T &length) const {
    SegmentLengthAndStart<T> out;
    out.length = length;
    out.start = 0;
    return out;
  }
};

int main() {
  size_t len = 5;

  thrust::host_vector<int> h_lengths(len);
  h_lengths[0] = 1;
  h_lengths[1] = 5;
  h_lengths[2] = 2;
  h_lengths[3] = 3;
  h_lengths[4] = 1;
  thrust::device_vector<int> d_lengths = h_lengths;

  thrust::device_vector<SegmentLengthAndStart<int>> d_out(len);

  auto inputIter = thrust::make_transform_iterator(
      d_lengths.begin(), SegmentLengthAndStartTransformer<int>());

  auto outputIter = d_out.begin();

  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, inputIter,
                                outputIter, len);
  thrust::device_vector<char> d_temp_storage(temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                inputIter, outputIter, len);

  thrust::host_vector<SegmentLengthAndStart<int>> h_out = d_out;
  for (size_t i = 0; i < h_out.size(); ++i) {
    printf("%d ", h_out[i].length);
  }
  printf("\n");
  for (size_t i = 0; i < h_out.size(); ++i) {
    printf("%d ", h_out[i].start);
  }
  printf("\n-----\n");

  // Plain exclusive sum, without combining starts with lengths
  thrust::device_vector<int> d_starts(len);
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_lengths.begin(), d_starts.begin(), len);
  d_temp_storage.resize(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes, d_lengths.begin(), d_starts.begin(), len);
  thrust::host_vector<int> h_starts = d_starts;
  for (size_t i = 0; i < h_starts.size(); ++i) {
    printf("%d ", h_starts[i]);
  }
  printf("\n");

  return 0;
}

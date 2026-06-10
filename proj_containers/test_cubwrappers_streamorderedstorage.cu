#include "containers/cubwrappers.cuh"
#include "containers/streams.cuh"

#include "gtest/gtest.h"

#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T> struct LessThan {
  T value;
  __device__ __forceinline__ bool operator()(const T &x) const {
    return x < value;
  }
};

struct CustomMin {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b < a) ? b : a;
  }
};

TEST(CubwStreamOrdered, DeviceRadixSortSortKeys) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_in{5, 3, 1, 4, 2};
  thrust::device_vector<int> d_out(5);

  cubw::DeviceRadixSort::SortKeys<int, int, true> sk(5, stream());
  sk.exec(d_in.data().get(), d_out.data().get(), 5, stream());

  stream.sync();
  thrust::host_vector<int> h = d_out;
  ASSERT_EQ(h[0], 1);
  ASSERT_EQ(h[1], 2);
  ASSERT_EQ(h[2], 3);
  ASSERT_EQ(h[3], 4);
  ASSERT_EQ(h[4], 5);
}

TEST(CubwStreamOrdered, DeviceMergeSortSortKeysCopy) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_in{5, 3, 1, 4, 2};
  thrust::device_vector<int> d_out(5);

  cubw::DeviceMergeSort::SortKeysCopy<int *, int *, int,
                                      ::cuda::std::less<int>, true>
      sk(5, stream());
  sk.exec(d_in.data().get(), d_out.data().get(), 5,
          ::cuda::std::less<int>{}, stream());

  stream.sync();
  thrust::host_vector<int> h = d_out;
  ASSERT_EQ(h[0], 1);
  ASSERT_EQ(h[1], 2);
  ASSERT_EQ(h[2], 3);
  ASSERT_EQ(h[3], 4);
  ASSERT_EQ(h[4], 5);
}

TEST(CubwStreamOrdered, DeviceMergeSortSortPairsCopy) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_keys_in{5, 3, 1, 4, 2};
  thrust::device_vector<float> d_vals_in{50.f, 30.f, 10.f, 40.f, 20.f};
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<float> d_vals_out(5);

  cubw::DeviceMergeSort::SortPairsCopy<int *, float *, int *, float *, int,
                                       ::cuda::std::less<int>, true>
      sp(5, stream());
  sp.exec(d_keys_in.data().get(), d_vals_in.data().get(),
          d_keys_out.data().get(), d_vals_out.data().get(), 5,
          ::cuda::std::less<int>{}, stream());

  stream.sync();
  thrust::host_vector<int> h_keys = d_keys_out;
  thrust::host_vector<float> h_vals = d_vals_out;
  ASSERT_EQ(h_keys[0], 1);
  ASSERT_EQ(h_keys[4], 5);
  ASSERT_FLOAT_EQ(h_vals[0], 10.f);
  ASSERT_FLOAT_EQ(h_vals[4], 50.f);
}

TEST(CubwStreamOrdered, DeviceSelectIfInPlace) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_data{1, 6, 2, 7, 3, 8, 4, 9};
  thrust::device_vector<int> d_num_selected(1);

  LessThan<int> functor{5};
  cubw::DeviceSelect::IfInPlace<int *, int *, LessThan<int>, true> sel(
      8, stream());
  sel.exec(d_data.data().get(), d_num_selected.data().get(), 8, functor,
           stream());

  stream.sync();
  thrust::host_vector<int> h_num = d_num_selected;
  ASSERT_EQ(h_num[0], 4);

  thrust::host_vector<int> h_data = d_data;
  for (int i = 0; i < 4; ++i)
    ASSERT_LT(h_data[i], 5);
}

TEST(CubwStreamOrdered, DeviceSelectIf) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_in{1, 6, 2, 7, 3, 8, 4, 9};
  thrust::device_vector<int> d_out(8);
  thrust::device_vector<int> d_num_selected(1);

  LessThan<int> functor{5};
  cubw::DeviceSelect::If<int *, int *, int *, LessThan<int>, true> sel(
      8, stream());
  sel.exec(d_in.data().get(), d_out.data().get(), d_num_selected.data().get(),
           8, functor, stream());

  stream.sync();
  thrust::host_vector<int> h_num = d_num_selected;
  ASSERT_EQ(h_num[0], 4);

  thrust::host_vector<int> h_out = d_out;
  ASSERT_EQ(h_out[0], 1);
  ASSERT_EQ(h_out[1], 2);
  ASSERT_EQ(h_out[2], 3);
  ASSERT_EQ(h_out[3], 4);
}

TEST(CubwStreamOrdered, DeviceScanExclusiveSumInPlace) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_inout{1, 1, 2, 2, 2, 3, 1, 1};

  cubw::DeviceScan::ExclusiveSumInPlace<int *, int, true> scan(8, stream());
  scan.exec(d_inout.data().get(), 8, stream());

  stream.sync();
  thrust::host_vector<int> h = d_inout;
  ASSERT_EQ(h[0], 0);
  ASSERT_EQ(h[1], 1);
  ASSERT_EQ(h[2], 2);
  ASSERT_EQ(h[3], 4);
  ASSERT_EQ(h[4], 6);
  ASSERT_EQ(h[5], 8);
  ASSERT_EQ(h[6], 11);
  ASSERT_EQ(h[7], 12);
}

TEST(CubwStreamOrdered, DeviceScanExclusiveSum) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_in{1, 1, 2, 2, 2, 3, 1, 1};
  thrust::device_vector<int> d_out(8);

  cubw::DeviceScan::ExclusiveSum<int *, int *, int, true> scan(8, stream());
  scan.exec(d_in.data().get(), d_out.data().get(), 8, stream());

  stream.sync();
  thrust::host_vector<int> h = d_out;
  ASSERT_EQ(h[0], 0);
  ASSERT_EQ(h[1], 1);
  ASSERT_EQ(h[2], 2);
  ASSERT_EQ(h[3], 4);
  ASSERT_EQ(h[4], 6);
  ASSERT_EQ(h[5], 8);
  ASSERT_EQ(h[6], 11);
  ASSERT_EQ(h[7], 12);
}

TEST(CubwStreamOrdered, DeviceScanInclusiveSumByKey) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_keys{0, 0, 0, 0, 1, 1, 1, 1};
  thrust::device_vector<int> d_vals{1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<int> d_out(8);

  // Must explicitly specify defaults to reach StreamOrdered
  cubw::DeviceScan::InclusiveSumByKey<int *, int *, int *,
                                      ::cuda::std::equal_to<>, uint32_t, true>
      scan(8, stream());
  scan.exec(d_keys.data().get(), d_vals.data().get(), d_out.data().get(), 8,
            ::cuda::std::equal_to<>{}, stream());

  stream.sync();
  thrust::host_vector<int> h = d_out;
  // Group 0: 1, 3, 6, 10
  ASSERT_EQ(h[0], 1);
  ASSERT_EQ(h[1], 3);
  ASSERT_EQ(h[2], 6);
  ASSERT_EQ(h[3], 10);
  // Group 1: 5, 11, 18, 26
  ASSERT_EQ(h[4], 5);
  ASSERT_EQ(h[5], 11);
  ASSERT_EQ(h[6], 18);
  ASSERT_EQ(h[7], 26);
}

TEST(CubwStreamOrdered, DeviceScanInclusiveScanByKey) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_keys{0, 0, 0, 0, 1, 1, 1, 1};
  thrust::device_vector<int> d_vals{8, 6, 7, 5, 3, 0, 9, 8};
  thrust::device_vector<int> d_out(8);

  cubw::DeviceScan::InclusiveScanByKey<int *, int *, int *, CustomMin,
                                       ::cuda::std::equal_to<>, uint32_t, true>
      scan(8, stream());
  scan.exec(d_keys.data().get(), d_vals.data().get(), d_out.data().get(),
            CustomMin{}, 8, ::cuda::std::equal_to<>{}, stream());

  stream.sync();
  thrust::host_vector<int> h = d_out;
  // Group 0: running min of {8, 6, 7, 5} -> {8, 6, 6, 5}
  ASSERT_EQ(h[0], 8);
  ASSERT_EQ(h[1], 6);
  ASSERT_EQ(h[2], 6);
  ASSERT_EQ(h[3], 5);
  // Group 1: running min of {3, 0, 9, 8} -> {3, 0, 0, 0}
  ASSERT_EQ(h[4], 3);
  ASSERT_EQ(h[5], 0);
  ASSERT_EQ(h[6], 0);
  ASSERT_EQ(h[7], 0);
}

TEST(CubwStreamOrdered, DeviceSegmentedReduceReduce) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_in{3, 1, 4, 1, 5, 9, 2, 6};
  thrust::device_vector<int> d_out(2);
  thrust::device_vector<int> d_begin{0, 4};
  thrust::device_vector<int> d_end{4, 8};

  cubw::DeviceSegmentedReduce::Reduce<int *, int *, int *, int *, CustomMin,
                                      int, true>
      red(2, stream());
  red.exec(d_in.data().get(), d_out.data().get(), 2, d_begin.data().get(),
           d_end.data().get(), CustomMin{}, 1000, stream());

  stream.sync();
  thrust::host_vector<int> h = d_out;
  ASSERT_EQ(h[0], 1); // min of {3,1,4,1}
  ASSERT_EQ(h[1], 2); // min of {5,9,2,6}
}

TEST(CubwStreamOrdered, DeviceRunLengthEncodeEncode) {
  containers::CudaStream stream;
  thrust::device_vector<int> d_in{1, 1, 2, 2, 2, 3, 1, 1};
  unsigned int numItems = 8;

  thrust::device_vector<int> d_unique(numItems);
  thrust::device_vector<unsigned int> d_lengths(numItems);
  thrust::device_vector<unsigned int> d_num_runs(1);

  cubw::DeviceRunLengthEncode::Encode<int *, int *, unsigned int *,
                                      unsigned int *, unsigned int, true>
      rle(numItems, stream());
  rle.exec(d_in.data().get(), d_unique.data().get(), d_lengths.data().get(),
           d_num_runs.data().get(), numItems, stream());

  stream.sync();
  thrust::host_vector<int> h_unique = d_unique;
  thrust::host_vector<unsigned int> h_lengths = d_lengths;
  thrust::host_vector<unsigned int> h_num_runs = d_num_runs;
  ASSERT_EQ(h_num_runs[0], 4u);
  ASSERT_EQ(h_unique[0], 1);
  ASSERT_EQ(h_unique[1], 2);
  ASSERT_EQ(h_unique[2], 3);
  ASSERT_EQ(h_unique[3], 1);
  ASSERT_EQ(h_lengths[0], 2u);
  ASSERT_EQ(h_lengths[1], 3u);
  ASSERT_EQ(h_lengths[2], 1u);
  ASSERT_EQ(h_lengths[3], 2u);
}

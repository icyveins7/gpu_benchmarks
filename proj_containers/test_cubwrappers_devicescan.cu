#include "containers/cubwrappers.cuh"

#include "gtest/gtest.h"

#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

TEST(ContainersCubw, DeviceScanExclusiveSumInPlace) {
  thrust::device_vector<int> d_inout{1, 1, 2, 2, 2, 3, 1, 1};

  cubw::DeviceScan::ExclusiveSumInPlace<int *, int> scan(8);
  scan.exec(d_inout.data().get(), d_inout.size());

  thrust::host_vector<int> h_inout = d_inout;

  ASSERT_EQ(h_inout[0], 0);
  ASSERT_EQ(h_inout[1], 1);
  ASSERT_EQ(h_inout[2], 2);
  ASSERT_EQ(h_inout[3], 4);
  ASSERT_EQ(h_inout[4], 6);
  ASSERT_EQ(h_inout[5], 8);
  ASSERT_EQ(h_inout[6], 11);
  ASSERT_EQ(h_inout[7], 12);
}

TEST(ContainersCubw, DeviceScanExclusiveSum) {
  thrust::device_vector<int> d_in{1, 1, 2, 2, 2, 3, 1, 1};
  thrust::device_vector<int> d_out(d_in.size());

  cubw::DeviceScan::ExclusiveSum<int *, int *, int> scan(8);
  scan.exec(d_in.data().get(), d_out.data().get(), d_in.size());

  thrust::host_vector<int> h_out = d_out;

  ASSERT_EQ(h_out[0], 0);
  ASSERT_EQ(h_out[1], 1);
  ASSERT_EQ(h_out[2], 2);
  ASSERT_EQ(h_out[3], 4);
  ASSERT_EQ(h_out[4], 6);
  ASSERT_EQ(h_out[5], 8);
  ASSERT_EQ(h_out[6], 11);
  ASSERT_EQ(h_out[7], 12);
}

TEST(ContainersCubw, DeviceScanExclusiveScanByKey) {
  // Two segments keyed by 0 and 1; exclusive prefix-sum within each segment.
  // keys:   {0, 0, 0, 1, 1, 1}
  // values: {1, 2, 3, 4, 5, 6}
  // expected output: {0, 1, 3, 0, 4, 9}
  thrust::device_vector<int> d_keys{0, 0, 0, 1, 1, 1};
  thrust::device_vector<int> d_values_in{1, 2, 3, 4, 5, 6};
  thrust::device_vector<int> d_values_out(d_values_in.size());

  using ScanOp = ::cuda::std::plus<int>;
  cubw::DeviceScan::ExclusiveScanByKey<int *, int *, int *, ScanOp, int> scan(6);
  scan.exec(d_keys.data().get(), d_values_in.data().get(),
            d_values_out.data().get(), ScanOp{}, 0, (uint32_t)d_values_in.size());

  thrust::host_vector<int> h_out = d_values_out;

  ASSERT_EQ(h_out[0], 0);
  ASSERT_EQ(h_out[1], 1);
  ASSERT_EQ(h_out[2], 3);
  ASSERT_EQ(h_out[3], 0);
  ASSERT_EQ(h_out[4], 4);
  ASSERT_EQ(h_out[5], 9);
}

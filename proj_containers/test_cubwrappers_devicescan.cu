#include "containers/cubwrappers.cuh"

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

TEST(ContainersCubw, DeviceScanExclusiveSum) {
  thrust::device_vector<int> d_inout{1, 1, 2, 2, 2, 3, 1, 1};

  cubw::DeviceScan::ExclusiveSum<int *, int> scan(8);
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

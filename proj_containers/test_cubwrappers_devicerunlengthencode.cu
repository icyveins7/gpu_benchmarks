#include "containers/cubwrappers.cuh"

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

TEST(ContainersCubw, DeviceRunLengthEncode_int) {

  thrust::device_vector<int> d_in{1, 1, 2, 2, 2, 3, 1, 1};
  unsigned int numItems = 8;

  cubw::DeviceRunLengthEncode::Encode<int *, int *, unsigned int *,
                                      unsigned int *, unsigned int>
      rle(numItems);

  thrust::device_vector<int> d_unique(numItems);
  thrust::device_vector<unsigned int> d_lengths(numItems);
  thrust::device_vector<unsigned int> d_num_runs(1);
  rle.exec(d_in.data().get(), d_unique.data().get(), d_lengths.data().get(),
           d_num_runs.data().get(), numItems);

  thrust::host_vector<int> h_unique = d_unique;
  thrust::host_vector<unsigned int> h_lengths = d_lengths;
  thrust::host_vector<unsigned int> h_num_runs = d_num_runs;
  ASSERT_EQ(h_unique[0], 1);
  ASSERT_EQ(h_unique[1], 2);
  ASSERT_EQ(h_unique[2], 3);
  ASSERT_EQ(h_unique[3], 1);
  for (int i = 4; i < 8; i++)
    ASSERT_EQ(h_unique[i], 0);
  ASSERT_EQ(h_lengths[0], 2);
  ASSERT_EQ(h_lengths[1], 3);
  ASSERT_EQ(h_lengths[2], 1);
  ASSERT_EQ(h_lengths[3], 2);
  for (int i = 4; i < 8; i++)
    ASSERT_EQ(h_lengths[i], 0);
  ASSERT_EQ(h_num_runs[0], 4);
}

TEST(ContainersCubw, DeviceRunLengthEncode_bool) {

  thrust::device_vector<bool> d_in{true,  true,  false, false,
                                   false, false, true,  true};
  unsigned int numItems = 8;

  cubw::DeviceRunLengthEncode::Encode<bool *, bool *, unsigned int *,
                                      unsigned int *, unsigned int>
      rle(numItems);

  thrust::device_vector<bool> d_unique(numItems);
  thrust::device_vector<unsigned int> d_lengths(numItems);
  thrust::device_vector<unsigned int> d_num_runs(1);
  rle.exec(d_in.data().get(), d_unique.data().get(), d_lengths.data().get(),
           d_num_runs.data().get(), numItems);

  thrust::host_vector<bool> h_unique = d_unique;
  thrust::host_vector<unsigned int> h_lengths = d_lengths;
  thrust::host_vector<unsigned int> h_num_runs = d_num_runs;

  ASSERT_EQ(h_unique[0], true);
  ASSERT_EQ(h_unique[1], false);
  ASSERT_EQ(h_unique[2], true);
  for (int i = 3; i < 8; i++)
    ASSERT_EQ(h_unique[i], false);
  ASSERT_EQ(h_lengths[0], 2);
  ASSERT_EQ(h_lengths[1], 4);
  ASSERT_EQ(h_lengths[2], 2);
  for (int i = 3; i < 8; i++)
    ASSERT_EQ(h_lengths[i], false);
  ASSERT_EQ(h_num_runs[0], 3);
}

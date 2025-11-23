#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>

#include "copy_ranges.cuh"

int main() {
  printf("Run length encoding and decoding\n");

  // Fixed length test
  // thrust::host_vector<int> h_keys = {0, 0, 3, 3, 3, 4, 4, 6, 7, 7};
  // Long vector test
  thrust::host_vector<int> h_keys(1000000);
  for (auto &key : h_keys) {
    key = std::rand() % 10000;
  }
  std::sort(h_keys.begin(), h_keys.end());

  thrust::device_vector<int> d_keys = h_keys;

  thrust::device_vector<int> d_unique_out(d_keys.size());
  thrust::device_vector<int> d_counts_out(d_keys.size());
  thrust::device_vector<int> d_num_runs_out(1);

  size_t temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(
      nullptr, temp_storage_bytes,
      (int *)nullptr, // d_keys.data().get(),
      (int *)nullptr, // d_unique_out.data().get(),
      (int *)nullptr, // d_counts_out.data().get(),
      d_num_runs_out.data().get(), d_keys.size());
  printf("temp_storage_bytes = %zu\n", temp_storage_bytes);

  thrust::device_vector<char> d_temp_storage(temp_storage_bytes);

  for (int iter = 0; iter < 3; ++iter) {
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage.data().get(), temp_storage_bytes, d_keys.data().get(),
        d_unique_out.data().get(), d_counts_out.data().get(),
        d_num_runs_out.data().get(), d_keys.size());
  }

  thrust::host_vector<int> h_unique_out = d_unique_out;
  thrust::host_vector<int> h_counts_out = d_counts_out;
  thrust::host_vector<int> h_num_runs_out = d_num_runs_out;

  printf("h_unique_out: ");
  for (int i = 0; i < 10; ++i) {
    printf("%d ", h_unique_out[i]);
  }
  printf("\n");
  printf("h_counts_out: ");
  for (int i = 0; i < 10; ++i) {
    printf("%d ", h_counts_out[i]);
  }
  printf("\n");
  printf("h_num_runs_out: %d\n", h_num_runs_out[0]);

  // Now attempt to decode
  // First we need to get the offsets via exclusive scan
  thrust::device_vector<int> d_offsets(h_num_runs_out[0]);

  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                d_counts_out.data().get(),
                                d_offsets.data().get(), h_num_runs_out[0]);
  if (temp_storage_bytes > d_temp_storage.size()) {
    d_temp_storage.resize(temp_storage_bytes);
    printf("Resized temp_storage to %zu for exclusive sum\n",
           temp_storage_bytes);
  }
  cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(), temp_storage_bytes,
                                d_counts_out.data().get(),
                                d_offsets.data().get(), h_num_runs_out[0]);

  thrust::host_vector<int> h_offsets = d_offsets;
  if (h_offsets.size() < 100) {
    for (size_t i = 0; i < h_offsets.size(); ++i) {
      printf("h_offsets[%zu] = %d\n", i, h_offsets[i]);
    }
  }

  // Then we use device copy batched
  thrust::device_vector<int> d_decoded(h_keys.size());
  thrust::device_vector<unsigned long long int> d_workcounter(1);

  // copy_ranges_blockwise<<<h_num_runs_out[0], 64>>>(
  copy_ranges_blockwise_worksteal<<<h_num_runs_out[0], 64>>>(
      d_keys.data().get(),       // input
      d_offsets.data().get(),    // offsets
      d_counts_out.data().get(), // lengths
      d_decoded.data().get(),    // output
      h_num_runs_out[0]          // num segments
      ,
      d_workcounter.data().get() // only for worksteal kernel
  );

  thrust::host_vector<int> h_decoded = d_decoded;
  for (size_t i = 0; i < 10; ++i) {
    int offset = h_offsets[i];
    printf("key[%zu]: length %d\n", i, h_counts_out[i]);
    printf("  orig  : ");
    for (size_t j = 0; j < 10; ++j) {
      printf("%d ", h_keys[offset + j]);
    }
    printf("...\n");

    printf(" decoded: ");
    for (size_t j = 0; j < 10; ++j) {
      printf("%d ", h_decoded[offset + j]);
    }
    printf("...\n");
  }

  for (size_t i = 0; i < h_decoded.size(); ++i) {
    if (h_decoded[i] != h_keys[i]) {
      printf("ERROR: decoded[%zu] = %d, expected %d\n", i, h_decoded[i],
             h_keys[i]);
    }
  }

  return 0;
}

// ============= THIS DOESNT EVEN FUCKING COMPILE ============

// int main() {
//   // https://github.com/NVIDIA/cccl/issues/599
//   const int num_segments = 4;
//   const int num_items = 12;
//   thrust::device_vector<int> segment_offsets = {0, 2, 8, num_items};
//   thrust::device_vector<int> out(num_items);
//   int *d_out = thrust::raw_pointer_cast(out.data());
//   int *d_offsets = thrust::raw_pointer_cast(segment_offsets.data());
//
//   thrust::counting_iterator<int> iota(0);
//   auto d_range_srcs = //
//       thrust::make_transform_iterator(
//           iota, [](int i) { return thrust::constant_iterator<int>(i); });
//
//   auto d_range_dsts = //
//       thrust::make_transform_iterator(
//           d_offsets, [d_out](int offset) { return d_out + offset; });
//
//   auto d_range_sizes = //
//       thrust::make_transform_iterator(
//           iota, [d_offsets](int i) { return d_offsets[i + 1] -
//           d_offsets[i];
//           });
//
//   std::uint8_t *d_temp_storage = nullptr;
//   std::size_t temp_storage_bytes = 0;
//   cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes,
//   d_range_srcs,
//                            d_range_dsts, d_range_sizes, num_segments);
//
//   return 0;
// }

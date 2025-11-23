#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>
//
// int main() {
//   printf("Run length encoding and decoding\n");
//
//   // Fixed length test
//   // thrust::host_vector<int> h_keys = {0, 0, 3, 3, 3, 4, 4, 6, 7, 7};
//   // Long vector test
//   thrust::host_vector<int> h_keys(1000000);
//   for (auto &key : h_keys) {
//     key = std::rand() % 10;
//   }
//   std::sort(h_keys.begin(), h_keys.end());
//
//   thrust::device_vector<int> d_keys = h_keys;
//
//   thrust::device_vector<int> d_unique_out(d_keys.size());
//   thrust::device_vector<int> d_counts_out(d_keys.size());
//   thrust::device_vector<int> d_num_runs_out(1);
//
//   size_t temp_storage_bytes = 0;
//   cub::DeviceRunLengthEncode::Encode(
//       nullptr, temp_storage_bytes,
//       (int *)nullptr, // d_keys.data().get(),
//       (int *)nullptr,
//       (int *)nullptr, // d_unique_out.data().get(),
//       d_counts_out.data().get(), (int *)nullptr, //
//       d_num_runs_out.data().get(), d_keys.size());
//   printf("temp_storage_bytes = %zu\n", temp_storage_bytes);
//
//   thrust::device_vector<char> d_temp_storage(temp_storage_bytes);
//
//   for (int iter = 0; iter < 3; ++iter) {
//     cub::DeviceRunLengthEncode::Encode(
//         d_temp_storage.data().get(), temp_storage_bytes, d_keys.data().get(),
//         d_unique_out.data().get(), d_counts_out.data().get(),
//         d_num_runs_out.data().get(), d_keys.size());
//   }
//
//   thrust::host_vector<int> h_unique_out = d_unique_out;
//   thrust::host_vector<int> h_counts_out = d_counts_out;
//   thrust::host_vector<int> h_num_runs_out = d_num_runs_out;
//
//   printf("h_unique_out: ");
//   for (int i = 0; i < 10; ++i) {
//     printf("%d ", h_unique_out[i]);
//   }
//   printf("\n");
//   printf("h_counts_out: ");
//   for (int i = 0; i < 10; ++i) {
//     printf("%d ", h_counts_out[i]);
//   }
//   printf("\n");
//   printf("h_num_runs_out: %d\n", h_num_runs_out[0]);
//
//   // Now attempt to decode
//   // First we need to get the offsets via exclusive scan
//   thrust::device_vector<int> d_offsets(h_num_runs_out[0]);
//
//   cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
//                                 d_counts_out.data().get(),
//                                 d_offsets.data().get(), h_num_runs_out[0]);
//   if (temp_storage_bytes > d_temp_storage.size()) {
//     d_temp_storage.resize(temp_storage_bytes);
//     printf("Resized temp_storage to %zu for exclusive sum\n",
//            temp_storage_bytes);
//   }
//   cub::DeviceScan::ExclusiveSum(d_temp_storage.data().get(),
//   temp_storage_bytes,
//                                 d_counts_out.data().get(),
//                                 d_offsets.data().get(), h_num_runs_out[0]);
//
//   thrust::host_vector<int> h_offsets = d_offsets;
//   for (size_t i = 0; i < h_offsets.size(); ++i) {
//     printf("h_offsets[%zu] = %d\n", i, h_offsets[i]);
//   }
//
//   // Then we use device copy batched
//   thrust::device_vector<int> d_decoded(h_keys.size());
//
//   // To do this we need to get pointers to each run using the counts (from
//   // original encode) and the offsets (from the scan)
//   struct GetIteratorForSrc {
//     __host__ __device__ __forceinline__ auto operator()(int i) const {
//       return thrust::make_constant_iterator(d_vec[i]);
//     }
//     int *d_vec;
//   };
//   struct GetPointerToRange {
//     __host__ __device__ __forceinline__ int *operator()(int i) const {
//       size_t offset = d_offsets[i]; // offset of the i'th run
//       return d_vec + offset;        // pointer to the start of i'th run
//     }
//     int *d_offsets;
//     int *d_vec;
//   };
//   // Similarly for the lengths
//   struct GetLength {
//     __host__ __device__ __forceinline__ int operator()(int index) const {
//       return d_counts_out[index];
//     }
//     int *d_counts_out;
//   };
//
//   int *d_offset_ptr = thrust::raw_pointer_cast(d_offsets.data());
//   int *d_keys_ptr = thrust::raw_pointer_cast(d_keys.data());
//   int *d_decoded_ptr = thrust::raw_pointer_cast(d_decoded.data());
//   int *d_counts_out_ptr = thrust::raw_pointer_cast(d_counts_out.data());
//
//   // auto input_ranges_iter = thrust::make_transform_iterator(
//   //     thrust::make_counting_iterator(0),
//   //     GetPointerToRangeConst{d_offset_ptr, d_keys_ptr}); // input ranges
//   auto input_ranges_iter = thrust::make_transform_iterator(
//       thrust::make_counting_iterator(0),
//       GetIteratorForSrc{d_keys_ptr}); // input ranges
//   auto output_ranges_iter = thrust::make_transform_iterator(
//       thrust::make_counting_iterator(0),
//       GetPointerToRange{d_offset_ptr, d_decoded_ptr}); // output ranges
//   auto lengths_iter =
//       thrust::make_transform_iterator(thrust::make_counting_iterator(0),
//                                       GetLength{d_counts_out_ptr}); //
//                                       lengths
//   cub::DeviceCopy::Batched(nullptr, temp_storage_bytes, input_ranges_iter,
//                            output_ranges_iter, lengths_iter,
//                            h_num_runs_out[0]);
//
//   // cub::DeviceCopy::Batched(
//   //     nullptr, temp_storage_bytes,
//   //     thrust::make_transform_iterator(
//   //         thrust::make_counting_iterator(0),
//   //         GetPointerToRange{d_offset_ptr, d_keys_ptr}), // input ranges
//   //     thrust::make_transform_iterator(
//   //         thrust::make_counting_iterator(0),
//   //         GetPointerToRange{d_offset_ptr, d_decoded_ptr}), // output
//   ranges
//   //     thrust::make_transform_iterator(thrust::make_counting_iterator(0),
//   //                                     GetLength{d_counts_out_ptr}), //
//   //                                     lengths
//   //     h_num_runs_out[0]);
//
//   if (temp_storage_bytes > d_temp_storage.size()) {
//     d_temp_storage.resize(temp_storage_bytes);
//     printf("Resized temp_storage to %zu for copy batched\n",
//            temp_storage_bytes);
//   }
//   //
//   // cub::DeviceCopy::Batched(
//   //     d_temp_storage.data().get(), temp_storage_bytes,
//   //     thrust::make_transform_iterator(
//   //         thrust::make_counting_iterator(0),
//   //         GetPointerToRange{d_offsets.data().get()}), // input ranges
//   //     thrust::make_transform_iterator(
//   //         thrust::make_counting_iterator(0),
//   //         GetPointerToRange{d_decoded.data().get()}), // output ranges
//   //     thrust::make_transform_iterator(
//   //         thrust::make_counting_iterator(0),
//   //         GetLength{d_counts_out.data().get()}), // lengths
//   //     h_num_runs_out[0]);
//
//   return 0;
// }

// ============= THIS DOESNT EVEN FUCKING COMPILE ============

int main() {
  // https://github.com/NVIDIA/cccl/issues/599
  const int num_segments = 4;
  const int num_items = 12;
  thrust::device_vector<int> segment_offsets = {0, 2, 8, num_items};
  thrust::device_vector<int> out(num_items);
  int *d_out = thrust::raw_pointer_cast(out.data());
  int *d_offsets = thrust::raw_pointer_cast(segment_offsets.data());

  thrust::counting_iterator<int> iota(0);
  auto d_range_srcs = //
      thrust::make_transform_iterator(
          iota, [](int i) { return thrust::constant_iterator<int>(i); });

  auto d_range_dsts = //
      thrust::make_transform_iterator(
          d_offsets, [d_out](int offset) { return d_out + offset; });

  auto d_range_sizes = //
      thrust::make_transform_iterator(
          iota, [d_offsets](int i) { return d_offsets[i + 1] - d_offsets[i]; });

  std::uint8_t *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, d_range_srcs,
                           d_range_dsts, d_range_sizes, num_segments);

  return 0;
}

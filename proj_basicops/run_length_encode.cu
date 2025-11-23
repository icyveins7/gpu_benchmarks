#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>

int main() {
  printf("Run length encoding\n");

  thrust::host_vector<int> h_keys = {0, 0, 3, 3, 3, 4, 4, 6, 7, 7};
  thrust::device_vector<int> d_keys = h_keys;

  thrust::device_vector<int> d_unique_out(10);
  thrust::device_vector<int> d_counts_out(10);
  thrust::device_vector<int> d_num_runs_out(1);

  size_t temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(
      nullptr, temp_storage_bytes,
      (int *)nullptr, // d_keys.data().get(),
      (int *)nullptr,
      (int *)nullptr, // d_unique_out.data().get(), d_counts_out.data().get(),
      (int *)nullptr, // d_num_runs_out.data().get(),
      d_keys.size());
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

  return 0;
}

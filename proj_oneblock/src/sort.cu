#include <iostream>

#include <thrust/device_vector.h>

#include <cub/cub.cuh>
#include <thrust/host_vector.h>

#include "containers/cubwrappers.cuh"

int main() {
  printf("One block sort comparisons\n");

  size_t temp_storage_bytes = 0;

  using KeyT = int;
  int num_items = 10000;
  printf("num_items: %d\n", num_items);
  thrust::host_vector<int> h_input(num_items);
  for (int i = 0; i < num_items; i++)
    h_input[i] = std::rand() % 1000;
  thrust::device_vector<int> d_input = h_input;
  thrust::device_vector<int> d_output(num_items);

  cubw::SortKeys<KeyT, int> sort_keys(num_items);
  sort_keys.exec(d_input.data().get(), d_output.data().get(), num_items);

  thrust::host_vector<int> h_output = d_output;
  for (int i = 1; i < num_items; i++) {
    if (h_output[i - 1] > h_output[i]) {
      std::cout << "Error at index " << i << std::endl;
    }
  }

  return 0;
}

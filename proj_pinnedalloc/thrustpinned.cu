#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

// my custom header to help keep everything in 1 place
#include "pinnedalloc.cuh"

int main() {
  const size_t len = 1000000;
  thrust::device_vector<double> d_x(len);
  thrust::sequence(d_x.begin(), d_x.end());

  // Create some host_vector without pinned allocator
  thrust::host_vector<double> h_x = d_x;

  // Now create with pinned allocator
  thrust::pinned_host_vector<double> h_px = d_x;

  for (size_t i = 0; i < len; i++) {
    if (h_x[i] != h_px[i]) {
      printf("[%zd]: %f vs %f\n", i, h_x[i], h_px[i]);
    }
  }
  printf("All ok\n");

  return 0;
}

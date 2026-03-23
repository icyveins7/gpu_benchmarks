#include <cstdint>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T, typename U>
__global__ void badprintfkernel(const T *a, const U *b) {
  printf("device: a = %d, b = %d, a = %d, a = %d\n", *a, *b, *a, *a);
  printf("device: a = %d, b (with ld) = %ld, a = %d\n", *a, *b, *a);
}

template <typename T, typename U> void printfhost(const T *a, const U *b) {
  printf("host: a = %d, b = %d, a = %d\n", *a, *b, *a);
  printf("host: a = %d, b (with ld) = %ld, a = %d\n", *a, *b, *a);
}

int main() {

  thrust::device_vector<int32_t> d_32(1);
  thrust::fill(d_32.begin(), d_32.end(), 0xFF332211);
  thrust::device_vector<int64_t> d_64(1);
  thrust::fill(d_64.begin(), d_64.end(), 0xFF332211112233FF);

  thrust::host_vector<int32_t> h_32 = d_32;
  thrust::host_vector<int64_t> h_64 = d_64;
  badprintfkernel<<<1, 1>>>(d_32.data().get(), d_64.data().get());
  printfhost(h_32.data(), h_64.data());

  return 0;
}

/*
 * This shows how printf behaviour is weird and corrupts subsequent arguments
 * when the format specifier is wrong, unlike host-side printf.
 *
 * host: a = -13426159, b = 287454207, a = -13426159
 * host: a = -13426159, b (with ld) = -57664913528441857, a = -13426159
 * device: a = -13426159, b = 0, a = 287454207, a = -13426159
 * device: a = -13426159, b (with ld) = -57664913528441857, a = -13426159
 */

#include "containers.cuh"
#include "timer.h"
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  size_t len = 8192 * 1024;

  // Create 1st vector
  std::vector<unsigned int> h_bitsetarrayvec(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));
  for (size_t i = 0; i < h_bitsetarrayvec.size(); ++i)
    h_bitsetarrayvec[i] = std::rand();

  thrust::device_vector<unsigned int> d_bitsetarrayvec(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));
  thrust::copy(h_bitsetarrayvec.begin(), h_bitsetarrayvec.end(),
               d_bitsetarrayvec.begin());
  containers::BitsetArray<unsigned int, int> d_bitsetarray(
      (containers::Bitset<unsigned int, int> *)d_bitsetarrayvec.data().get(),
      len);

  // Create 2nd vector
  std::vector<unsigned int> h_bitsetarrayvec2(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));
  for (size_t i = 0; i < h_bitsetarrayvec2.size(); ++i)
    h_bitsetarrayvec2[i] = std::rand();

  thrust::device_vector<unsigned int> d_bitsetarrayvec2(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));
  thrust::copy(h_bitsetarrayvec2.begin(), h_bitsetarrayvec2.end(),
               d_bitsetarrayvec2.begin());
  containers::BitsetArray<unsigned int, int> d_bitsetarray2(
      (containers::Bitset<unsigned int, int> *)d_bitsetarrayvec2.data().get(),
      len);

  // Create 3rd vector
  thrust::device_vector<unsigned int> d_bitsetarrayvec3(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));

  printf("Total data elements = %d\n", d_bitsetarray.numDataElements);

  thrust::device_vector<int> d_minIdx(1);
  thrust::device_vector<int> d_floor(1);

  thrust::host_vector<int> h_minIdx(1);

  for (int i = 0; i < 3; ++i) {
    // reset min index
    d_minIdx[0] = len;
    int TPB = 128;
    int NUM_BLKS = 64; //(d_bitsetarrayvec.size() + TPB - 1) / TPB;
    containers::argminBitGlobalKernel<<<NUM_BLKS, TPB>>>(d_bitsetarray,
                                                         d_minIdx.data().get());

    h_minIdx = d_minIdx;
    printf("%d\n", h_minIdx[0]);

    // reset min index
    d_minIdx[0] = len;
    d_floor[0] = len - 1000; // setting and using this makes it very fast
    containers::argminBitGlobalKernel<<<NUM_BLKS, TPB>>>(
        d_bitsetarray, d_minIdx.data().get(), d_floor.data().get());

    h_minIdx = d_minIdx;
    printf("%d\n", h_minIdx[0]);

    {
      // Rough comparison with CPU timing to iterate
      thrust::host_vector<containers::Bitset<unsigned int, int>>
          h_bitsetarrayvec = d_bitsetarrayvec;
      // HighResolutionTimer timer;
      // for (size_t i = 0; i < h_bitsetarrayvec.size(); ++i) {
      //   if (h_bitsetarrayvec[i].value != 0) {
      //     printf("%zu\n", i);
      //   }
      // }
    }

    // Test some bitwise operations
    {
      thrust::transform(d_bitsetarrayvec.begin(), d_bitsetarrayvec.end(),
                        d_bitsetarrayvec2.begin(), d_bitsetarrayvec3.begin(),
                        thrust::bit_or<unsigned int>());

      thrust::transform(d_bitsetarrayvec.begin(), d_bitsetarrayvec.end(),
                        d_bitsetarrayvec2.begin(), d_bitsetarrayvec3.begin(),
                        thrust::bit_and<unsigned int>());

      thrust::host_vector<unsigned int> h_check = d_bitsetarrayvec3;
      printf("%X & %X = %X\n", h_bitsetarrayvec[0], h_bitsetarrayvec2[0],
             h_check[0]);
    }
  }

  return 0;
}

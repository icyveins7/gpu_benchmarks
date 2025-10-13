#include "containers.cuh"
#include "timer.h"
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  size_t len = 8192 * 1024;
  thrust::device_vector<unsigned int> d_bitsetarrayvec(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));
  containers::BitsetArray<unsigned int, int> d_bitsetarray(
      (containers::Bitset<unsigned int, int> *)d_bitsetarrayvec.data().get(),
      len);

  thrust::device_vector<unsigned int> d_bitsetarrayvec2(
      containers::BitsetArray<unsigned int, int>::numElementsRequiredFor(len));
  containers::BitsetArray<unsigned int, int> d_bitsetarray2(
      (containers::Bitset<unsigned int, int> *)d_bitsetarrayvec2.data().get(),
      len);
  thrust::fill(d_bitsetarrayvec2.begin(), d_bitsetarrayvec2.end(), 1);

  printf("Total data elements = %d\n", d_bitsetarray.numDataElements);

  thrust::device_vector<int> d_minIdx(1);
  thrust::device_vector<int> d_floor(1);

  thrust::host_vector<int> h_minIdx(1);

  for (int i = 0; i < 3; ++i) {

    d_bitsetarrayvec[0] = 1;
    d_minIdx[0] = len;
    int TPB = 128;
    int NUM_BLKS = 64; //(d_bitsetarrayvec.size() + TPB - 1) / TPB;
    containers::argminBitGlobalKernel<<<NUM_BLKS, TPB>>>(d_bitsetarray,
                                                         d_minIdx.data().get());

    h_minIdx = d_minIdx;
    printf("%d\n", h_minIdx[0]);

    d_bitsetarrayvec[0] = 0;
    d_bitsetarrayvec[d_bitsetarrayvec.size() - 1] = 1;
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
                        d_bitsetarrayvec2.begin(), d_bitsetarrayvec.begin(),
                        thrust::bit_or<unsigned int>());

      thrust::transform(d_bitsetarrayvec.begin(), d_bitsetarrayvec.end(),
                        d_bitsetarrayvec2.begin(), d_bitsetarrayvec.begin(),
                        thrust::bit_and<unsigned int>());
    }
  }

  return 0;
}

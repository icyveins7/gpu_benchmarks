#include <cstdlib>
#include <iostream>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include "devicereduction.cuh"

int main(int argc, char **argv) {
  size_t length = 1024;
  if (argc >= 2) {
    length = atoi(argv[1]);
  }
  printf("Length: %zd\n", length);

  using AccumulatorType = unsigned long long;
  using BaseType = unsigned short;

  thrust::host_vector<BaseType> h_x(length);
  AccumulatorType checkSum = 0, checkSumSq = 0;
  for (size_t i = 0; i < length; ++i) {
    h_x[i] = std::rand() % 10000;
    checkSum += h_x[i];
    checkSumSq += h_x[i] * h_x[i];
  }
  thrust::device_vector<BaseType> d_x(h_x);

  // Call the kernel
  thrust::device_vector<AccumulatorType> d_sum(1);
  thrust::device_vector<AccumulatorType> d_sumSq(1);
  int TPB = 256;
  int NUM_BLKS = length / TPB + 1;
  device_sum_and_sumSq_kernel<BaseType, AccumulatorType, true>
      <<<NUM_BLKS, TPB, sizeof(AccumulatorType) * TPB>>>(
          d_x.data().get(), length, d_sum.data().get(), d_sumSq.data().get());
  device_sum_and_sumSq_kernel<BaseType, AccumulatorType, false>
      <<<NUM_BLKS, TPB>>>(d_x.data().get(), length, d_sum.data().get(),
                          d_sumSq.data().get());
  thrust::host_vector<AccumulatorType> h_sum(d_sum);
  thrust::host_vector<AccumulatorType> h_sumSq(d_sumSq);
  printf("Sum: %llu\n", h_sum[0] / 2);
  printf("SumSq: %llu\n", h_sumSq[0] / 2);
  printf("CheckSum: %llu\n", checkSum);
  printf("CheckSumSq: %llu\n", checkSumSq);

  // What if we called the sectioned flavour of the kernel for just this
  // instead? NOTE: this appears to be almost the same time as above, yay!
  device_sectioned_sum_and_sumSq_kernel<BaseType, AccumulatorType, size_t>
      <<<NUM_BLKS, TPB>>>(d_x.data().get(), length, d_sum.data().get(),
                          d_sumSq.data().get(), length, 1);
  h_sum = d_sum;
  h_sumSq = d_sumSq;

  printf("Sum: %llu\n", h_sum[0] / 3);
  printf("SumSq: %llu\n", h_sumSq[0] / 3);
  printf("CheckSum: %llu\n", checkSum);
  printf("CheckSumSq: %llu\n", checkSumSq);

  return 0;
}

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

  // Now try doing multiple sections
  const size_t sectionLength = 100000;
  const size_t numSections = 6;
  thrust::device_vector<AccumulatorType> d_sumSections(numSections);
  thrust::device_vector<AccumulatorType> d_sumSqSections(numSections);
  h_x.resize(sectionLength * numSections);
  AccumulatorType checkSums[numSections] = {0};
  AccumulatorType checkSumSqs[numSections] = {0};
  for (size_t i = 0; i < numSections; ++i) {
    for (size_t j = 0; j < sectionLength; ++j) {
      h_x[i * sectionLength + j] = std::rand() % 10000;
      checkSums[i] += h_x[i * sectionLength + j];
      checkSumSqs[i] += h_x[i * sectionLength + j] * h_x[i * sectionLength + j];
    }
  }
  d_x = h_x;

  // Call sectioned kernel
  NUM_BLKS = d_x.size() / TPB + 1;
  printf("Total threads in grid = %zd\n", (size_t)NUM_BLKS * (size_t)TPB);
  device_sectioned_sum_and_sumSq_kernel<BaseType, AccumulatorType, size_t>
      <<<NUM_BLKS, TPB>>>(
          d_x.data().get(), d_x.size(), d_sumSections.data().get(),
          d_sumSqSections.data().get(), sectionLength, numSections);

  thrust::host_vector<AccumulatorType> h_sumSections = d_sumSections;
  thrust::host_vector<AccumulatorType> h_sumSqSections = d_sumSqSections;
  for (size_t i = 0; i < numSections; ++i) {
    printf("--------- Section %zd\n", i);
    printf("Sum: %llu\n", h_sumSections[i]);
    printf("SumSq: %llu\n", h_sumSqSections[i]);
    printf("CheckSum: %llu\n", checkSums[i]);
    printf("CheckSumSq: %llu\n", checkSumSqs[i]);
    printf("%s\n", h_sumSections[i] == checkSums[i] &&
                           h_sumSqSections[i] == checkSumSqs[i]
                       ? "\u2713"
                       : "\u2715");
  }

  return 0;
}

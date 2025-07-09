#include <cstdlib>
#include <iostream>
#include <limits>

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

  {
    int TPB = 256;
    int NUM_BLKS = length / TPB + 1;
    device_sum_and_sumSq_kernel<BaseType, AccumulatorType, true>
        <<<NUM_BLKS, TPB, sizeof(AccumulatorType) * TPB>>>(
            d_x.data().get(), length, d_sum.data().get(), d_sumSq.data().get());
    device_sum_and_sumSq_kernel<BaseType, AccumulatorType, false>
        <<<NUM_BLKS, TPB>>>(d_x.data().get(), length, d_sum.data().get(),
                            d_sumSq.data().get());
  }
  thrust::host_vector<AccumulatorType> h_sum(d_sum);
  thrust::host_vector<AccumulatorType> h_sumSq(d_sumSq);
  printf("Sum: %llu\n", h_sum[0] / 2);
  printf("SumSq: %llu\n", h_sumSq[0] / 2);
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
  unsigned int checkValidSize[numSections] = {0};
  for (size_t i = 0; i < numSections; ++i) {
    for (size_t j = 0; j < sectionLength; ++j) {
      auto value = std::rand() % 10000;
      h_x[i * sectionLength + j] =
          value >= 9000 ? std::numeric_limits<BaseType>::max() : value;
      // printf("Section %zd index %zd = %hu\n", i, j, h_x[i * sectionLength +
      // j]);
      if (h_x[i * sectionLength + j] != std::numeric_limits<BaseType>::max()) {
        checkSums[i] += h_x[i * sectionLength + j];
        checkSumSqs[i] +=
            h_x[i * sectionLength + j] * h_x[i * sectionLength + j];
        checkValidSize[i]++;
      }
    }
  }
  d_x = h_x;

  thrust::device_vector<unsigned int> d_validSectionSizes(numSections);

  // Call sectioned kernel
  {
    int TPB(128);
    dim3 NUM_BLKS(sectionLength / TPB + 1, numSections, 1);
    device_sectioned_sum_and_sumSq_kernel<BaseType, AccumulatorType,
                                          unsigned int>
        <<<NUM_BLKS, TPB>>>(d_x.data().get(), d_sumSections.data().get(),
                            d_sumSqSections.data().get(), sectionLength,
                            numSections, d_validSectionSizes.data().get(),
                            cuda::std::numeric_limits<BaseType>::max());
  }

  thrust::host_vector<AccumulatorType> h_sumSections = d_sumSections;
  thrust::host_vector<AccumulatorType> h_sumSqSections = d_sumSqSections;
  thrust::host_vector<unsigned int> h_validSectionSizes = d_validSectionSizes;
  for (size_t i = 0; i < numSections; ++i) {
    printf("--------- Section %zd\n", i);
    printf("Valid length: %u\n", h_validSectionSizes[i]);
    printf("Sum         : %llu\n", h_sumSections[i]);
    printf("SumSq       : %llu\n", h_sumSqSections[i]);

    printf("Check Valid length: %u\n", checkValidSize[i]);
    printf("CheckSum    : %llu\n", checkSums[i]);
    printf("CheckSumSq  : %llu\n", checkSumSqs[i]);
    printf("%s\n", h_sumSections[i] == checkSums[i] &&
                           h_sumSqSections[i] == checkSumSqs[i] &&
                           h_validSectionSizes[i] == checkValidSize[i]
                       ? "\u2713"
                       : "\u2715");
  }

  return 0;
}

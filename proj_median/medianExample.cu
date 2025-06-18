#include "median.cuh"

#include "timer.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <limits>
#include <vector>

template <typename T> void testKernel(int numTests, int maxLength) {
  // Initialise inputs to all maximum values
  std::vector<T> input(maxLength * numTests, std::numeric_limits<T>::max());
  std::vector<int> inputLengths(numTests);

  // Randomise a few lengths
  for (int i = 0; i < numTests; i++) {
    int l = 0;
    // Don't want 0-length vectors
    while (l == 0)
      l = rand() % maxLength;

    inputLengths[i] = l;
  }

  // Now randomise values up to the length
  for (int i = 0; i < numTests; ++i) {
    for (int j = 0; j < inputLengths[i]; ++j)
      input[i * maxLength + j] = rand() % 1000;
  }
  printf("Randomised values.\n");

  // Run our kernel
  thrust::device_vector<T> d_input(input);
  thrust::device_vector<int> d_inputLengths(inputLengths);
  thrust::device_vector<T> d_medians(numTests);

  const int numThreads = 128;
  const int ELEM_PER_THREAD = 1;
  const int numBlocks = numTests;

  printf("Starting kernel, %d blocks, %d threads.\n", numBlocks, numThreads);
  blockwise_median_kernel<T, numThreads, ELEM_PER_THREAD>
      <<<numBlocks, numThreads>>>(d_input.data().get(), numTests, maxLength,
                                  d_inputLengths.data().get(),
                                  d_medians.data().get());
  printf("Kernel complete\n");

  thrust::host_vector<T> h_medians = d_medians;
  printf("Copied kernel results back\n");

  // Now run the original data with CPU nth_element
  std::vector<T> medianChecks(h_medians.size());
  {
    HighResolutionTimer timer;
    for (int i = 0; i < numTests; i++) {
      std::nth_element(input.begin() + i * maxLength,
                       input.begin() + i * maxLength + inputLengths[i] / 2,
                       input.begin() + i * maxLength + inputLengths[i]);
      medianChecks[i] = input[i * maxLength + inputLengths[i] / 2];
    }
  }

  for (int i = 0; i < numTests; i++) {
    if (h_medians[i] != medianChecks[i]) {
      std::cout << "Median check failed at index " << i << "!" << std::endl;
      std::cout << "Input length: " << inputLengths[i] << std::endl;
      std::cout << "Median, index " << i << ": " << h_medians[i] << " vs "
                << medianChecks[i] << std::endl;
    }
  }
}

int main() {
  printf("Median kernel tests\n");

  testKernel<unsigned short>(10000, 100);

  return 0;
}

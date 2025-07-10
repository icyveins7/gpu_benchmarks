#include <iostream>

#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../proj_reductions/devicereduction.cuh"
#include "outlierRemoval.cuh"

int main() {
  // Outlier removal test
  printf("Outlier removal via iterative kernels\n");

  // Note that due to the iterative nature, it is very hard to design a specific
  // example to demonstrate outlier removal in several steps. Here we do our
  // best to try to test this functionality so that it's not trivial (not just
  // removal in 1 step). This requires specific tuning of the std deviation
  // factor.

  float stdFactor = 1.52;

  unsigned int maxSectionSize = 10;
  unsigned int numSections = 3;
  thrust::host_vector<unsigned short> h_x = {
      0, 0, 5, 6, 6,
      5, 7, 5, 6, 9, // should drop 0s first, then drop the 9, then the 7
      9, 8, 9, 8, 9,
      8, 9, 8, 9, 8, // should not trigger any outlier removal
      1, 1, 1, 1, 1,
      1, 1, 1, 9, 9 // should only drop the 9s
  };
  thrust::device_vector<unsigned short> d_x = h_x;
  printf("Original data\n");
  for (unsigned int i = 0; i < numSections; ++i) {
    for (unsigned int j = 0; j < maxSectionSize; ++j) {
      printf("%hu ", h_x[i * maxSectionSize + j]);
    }
    printf("\n===========\n");
  }

  // Initialize section sizes
  thrust::device_vector<unsigned int> d_currSectionSizes(numSections);
  thrust::device_vector<unsigned int> d_prevSectionSizes(numSections);
  thrust::device_vector<unsigned int> d_nextSectionSizes(numSections);

  // Initialize sums
  thrust::device_vector<unsigned int> d_sumSections(numSections);
  thrust::device_vector<unsigned int> d_sumSqSections(numSections);
  {
    int TPB = 32;
    dim3 NUM_BLKS(maxSectionSize / TPB + 1, numSections, 1);
    device_sectioned_sum_and_sumSq_kernel<unsigned short, unsigned int,
                                          unsigned int>
        <<<NUM_BLKS, TPB>>>(d_x.data().get(), d_sumSections.data().get(),
                            d_sumSqSections.data().get(), maxSectionSize,
                            numSections, d_currSectionSizes.data().get());
  }
  // Copy curr size to next size
  d_nextSectionSizes = d_currSectionSizes;

  // Start iterations
  int numIters = 3;
  for (int i = 0; i < numIters; ++i) {
    printf("-----------------\nIter %d\n", i);
    // Call kernel
    {
      int TPB = 32;
      dim3 NUM_BLKS(maxSectionSize / TPB + 1, numSections, 1);
      remove_sigma_multiple_outlier_removal_sectioned_kernel<
          unsigned short, unsigned int, float, unsigned int><<<NUM_BLKS, TPB>>>(
          d_x.data().get(), maxSectionSize, d_currSectionSizes.data().get(),
          d_prevSectionSizes.data().get(), d_nextSectionSizes.data().get(),
          numSections, d_sumSections.data().get(), d_sumSqSections.data().get(),
          stdFactor);
    }
    // Copy curr size to prev size
    d_prevSectionSizes = d_currSectionSizes;
    // Copy next size to curr size
    d_currSectionSizes = d_nextSectionSizes;

    // To show how it works, we copy each round to host
    thrust::host_vector<unsigned int> h_currSectionSizes = d_currSectionSizes;
    thrust::host_vector<unsigned int> h_prevSectionSizes = d_prevSectionSizes;
    h_x = d_x;
    thrust::host_vector<unsigned int> h_sumSections = d_sumSections;
    thrust::host_vector<unsigned int> h_sumSqSections = d_sumSqSections;

    for (unsigned int s = 0; s < numSections; ++s) {
      printf("Section %u: valid size %u -> %u\n", s, h_prevSectionSizes[s],
             h_currSectionSizes[s]);
      for (unsigned int j = 0; j < maxSectionSize; ++j) {
        if (h_x[s * maxSectionSize + j] !=
            std::numeric_limits<unsigned short>::max())
          printf("%hu ", h_x[s * maxSectionSize + j]);
        else
          printf("* ");
      }
      printf("\n [Sum %u, SumSq %u]\n", h_sumSections[s], h_sumSqSections[s]);
    }
  }

  return 0;
}

#include <cmath>
#include <iostream>

#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../proj_median/median.cuh"
#include "../proj_reductions/devicereduction.cuh"
#include "outlierOverwriting.cuh"
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

  constexpr int maxSectionSize = 10;
  constexpr int numSections = 3;
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
  for (int i = 0; i < numSections; ++i) {
    for (int j = 0; j < maxSectionSize; ++j) {
      printf("%hu ", h_x[i * maxSectionSize + j]);
    }
    printf("\n===========\n");
  }

  // Initialize section sizes
  thrust::device_vector<int> d_currSectionSizes(numSections);
  thrust::device_vector<int> d_prevSectionSizes(numSections);
  thrust::device_vector<int> d_nextSectionSizes(numSections);
  thrust::host_vector<int> h_currSectionSizes(numSections);
  thrust::host_vector<int> h_prevSectionSizes(numSections);
  thrust::host_vector<int> h_nextSectionSizes(numSections);

  // Initialize sums
  thrust::device_vector<unsigned int> d_sumSections(numSections);
  thrust::device_vector<unsigned int> d_sumSqSections(numSections);
  {
    int TPB = 32;
    dim3 NUM_BLKS(maxSectionSize / TPB + 1, numSections, 1);
    device_sectioned_sum_and_sumSq_kernel<unsigned short, unsigned int, int>
        <<<NUM_BLKS, TPB>>>(d_x.data().get(), d_sumSections.data().get(),
                            d_sumSqSections.data().get(), maxSectionSize,
                            numSections, d_currSectionSizes.data().get());
  }
  // Copy curr size to next size
  d_nextSectionSizes = d_currSectionSizes;

  // Start iterations
  thrust::host_vector<unsigned int> h_sumSqSections(d_sumSqSections.size());
  thrust::host_vector<unsigned int> h_sumSections(d_sumSections.size());
  int numIters = 3;
  for (int i = 0; i < numIters; ++i) {
    printf("-----------------\nIter %d\n", i);
    // Call kernel
    {
      int TPB = 32;
      dim3 NUM_BLKS(maxSectionSize / TPB + 1, numSections, 1);
      remove_sigma_multiple_outlier_removal_sectioned_kernel<
          unsigned short, unsigned int, float, int><<<NUM_BLKS, TPB>>>(
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
    h_currSectionSizes = d_currSectionSizes;
    h_prevSectionSizes = d_prevSectionSizes;
    h_x = d_x;
    h_sumSections = d_sumSections;
    h_sumSqSections = d_sumSqSections;

    for (int s = 0; s < numSections; ++s) {
      printf("Section %u: valid size %u -> %u\n", s, h_prevSectionSizes[s],
             h_currSectionSizes[s]);
      for (int j = 0; j < maxSectionSize; ++j) {
        if (h_x[s * maxSectionSize + j] !=
            std::numeric_limits<unsigned short>::max())
          printf("%hu ", h_x[s * maxSectionSize + j]);
        else
          printf("* ");
      }
      printf(" [Sum %u, SumSq %u]\n", h_sumSections[s], h_sumSqSections[s]);
    }
  }

  // Now try to overwrite outliers
  // But first we need to estimate the medians from the remaining elements
  thrust::device_vector<unsigned short> d_medians(numSections);
  {
    int NUM_BLKS = numSections;
    constexpr int TPB = 32;
    constexpr int ELEM_PER_THREAD =
        maxSectionSize / TPB + (maxSectionSize % TPB == 0 ? 0 : 1);
    blockwise_median_kernel<unsigned short, TPB, ELEM_PER_THREAD, false>
        <<<NUM_BLKS, TPB>>>(d_x.data().get(), numSections, maxSectionSize,
                            d_currSectionSizes.data().get(),
                            d_medians.data().get());
  }
  thrust::host_vector<unsigned short> h_medians = d_medians;
  for (int s = 0; s < numSections; ++s) {
    float mean = (float)h_sumSections[s] / h_currSectionSizes[s];
    float var = (float)h_sumSqSections[s] / h_currSectionSizes[s] - mean * mean;
    float std = std::sqrt(var);
    printf("Section %u: median %u [thresholds %f - %f]\n", s, h_medians[s],
           h_medians[s] - stdFactor * std, h_medians[s] + stdFactor * std);
  }

  // Now overwrite all outliers in bigger sample of data
  // Pretend the small sections of data came from these
  int bigDataLength = 100;
  thrust::host_vector<unsigned short> h_bigdata(numSections * bigDataLength);
  for (int i = 0; i < h_bigdata.size(); ++i)
    h_bigdata[i] = std::rand() % 20;

  printf("===============================================\n");
  thrust::device_vector<unsigned short> d_bigdata(h_bigdata);

  {
    int NUM_THREADS = 32;
    dim3 NUM_BLKS(bigDataLength / NUM_THREADS + 1, numSections);
    minmax_hold_from_section_medians_thresholds_kernel<unsigned short,
                                                       unsigned int, float, int>
        <<<NUM_BLKS, NUM_THREADS>>>(d_bigdata.data().get(), bigDataLength,
                                    numSections, d_medians.data().get(),
                                    d_sumSections.data().get(),
                                    d_sumSqSections.data().get(),
                                    d_currSectionSizes.data().get(), stdFactor);
  }
  // Now let's look at the final data
  thrust::host_vector<unsigned short> h_bigdata_after = d_bigdata;
  for (int i = 0; i < numSections; ++i) {
    printf("Section %d\n", i);
    for (int j = 0; j < bigDataLength; ++j) {
      printf("  %2hu -> %2hu\n", h_bigdata[i * bigDataLength + j],
             h_bigdata_after[i * bigDataLength + j]);
    }
  }
  printf("===============================================\n");

  return 0;
}

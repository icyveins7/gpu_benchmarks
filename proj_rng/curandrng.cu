#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "curand_extensions.h"

int main(int argc, char *argv[]) {
  printf("cuRAND RNGs\n");

  size_t numPts = 1000000;
  if (argc >= 2)
    numPts = strtol(argv[1], nullptr, 10);
  printf("Using numPts = %ld\n", numPts);

  int repeats = 5;
  if (argc >= 3)
    repeats = strtol(argv[2], nullptr, 10);
  printf("Using repeats = %d\n", repeats);

  // Test the most basic cuRAND RNG
  {
    printf("Basic cuRAND RNG\n");
    thrust::device_vector<float> d_x(numPts);

    CuRandRNG<float> rng(0, numPts);

    for (int i = 0; i < repeats; ++i) {
      rng.rand(d_x);

      thrust::host_vector<float> h_x = d_x;

      for (int i = 0; i < 10; ++i) {
        printf("%.3f\n", h_x[i]);
      }
      printf("====================================\n");
    }
  }

  // Test the patch-wise batched cuRAND RNG
  {
    printf("Batched cuRAND RNG\n");
    const unsigned int patchWidth = 2, patchHeight = 2;
    const unsigned int oWidth = 4, oHeight = 4;
    const unsigned int batch = 4;
    thrust::device_vector<float> d_x(oWidth * oHeight * batch);

    CuRandRNGPatchInBatch<float> rng(0, batch, patchWidth, patchHeight);

    thrust::host_vector<float> h_x(d_x.size());
    for (int i = 0; i < repeats; ++i) {
      printf("Repeat %d\n", i);
      rng.rand(d_x, oWidth, oHeight);

      h_x = d_x;

      for (size_t b = 0; b < batch; ++b) {
        for (size_t y = 0; y < oHeight; ++y) {
          for (size_t x = 0; x < oWidth; ++x) {
            printf("%.3f ", h_x[b * oWidth * oHeight + y * oWidth + x]);
          }
          printf("\n");
        }
        printf("------------------------\n");
      }
      printf("====================================\n");
    }
    printf("rng offset is now %llu\n", rng.getOffset());

    // Before we end, we test how skipahead works

    printf("Creating identical RNG to test skipahead\n");
    CuRandRNGPatchInBatch<float> rng2(0, batch, patchWidth, patchHeight);
    thrust::device_vector<float> d_x2(d_x.size());

    // Skip ahead to the last one
    rng2.skipahead(repeats - 1);
    // next one will be repeats - 1,
    // which will be the same as the above one
    rng2.rand(d_x2, oWidth, oHeight);

    thrust::host_vector<float> h_x2 = d_x2;
    printf("Checking equality with non-skippedahead RNG\n");

    for (size_t b = 0; b < batch; ++b) {
      for (size_t y = 0; y < oHeight; ++y) {
        bool allOk = true;
        for (size_t x = 0; x < oWidth; ++x) {
          printf("%.3f ", h_x2[b * oWidth * oHeight + y * oWidth + x]);
          if (h_x[b * oWidth * oHeight + y * oWidth + x] !=
              h_x2[b * oWidth * oHeight + y * oWidth + x])
            allOk = false;
        }
        if (!allOk)
          printf(" FAILED CHECK ");
        printf("\n");
      }
      printf("------------------------\n");
    }
    printf("====================================\n");
    printf("rng offset is now %llu\n", rng2.getOffset());

    // we can also instantiate directly at that offset
    printf("Creating RNG with offset value directly: %llu\n",
           rng2.getOffset() - 1);
    CuRandRNGPatchInBatch<float> rng3(0, batch, patchWidth, patchHeight,
                                      rng2.getOffset() - 1);
    rng3.rand(d_x2, oWidth, oHeight);
    h_x2 = d_x2;
    printf("Checking equality with non-skippedahead RNG\n");
    for (size_t b = 0; b < batch; ++b) {
      for (size_t y = 0; y < oHeight; ++y) {
        bool allOk = true;
        for (size_t x = 0; x < oWidth; ++x) {
          printf("%.3f ", h_x2[b * oWidth * oHeight + y * oWidth + x]);
          if (h_x[b * oWidth * oHeight + y * oWidth + x] !=
              h_x2[b * oWidth * oHeight + y * oWidth + x])
            allOk = false;
        }
        if (!allOk)
          printf(" FAILED CHECK ");
        printf("\n");
      }
      printf("------------------------\n");
    }
    printf("====================================\n");

    // we can mix and match
    printf("Creating RNG with offset 1, and then skipping ahead %llu\n",
           rng2.getOffset() - 1 - 1);
    CuRandRNGPatchInBatch<float> rng4(0, batch, patchWidth, patchHeight, 1);
    rng4.skipahead(rng2.getOffset() - 1 - 1);
    rng4.rand(d_x2, oWidth, oHeight);
    h_x2 = d_x2;

    printf("Checking equality with non-skippedahead RNG\n");
    for (size_t b = 0; b < batch; ++b) {
      for (size_t y = 0; y < oHeight; ++y) {
        bool allOk = true;
        for (size_t x = 0; x < oWidth; ++x) {
          printf("%.3f ", h_x2[b * oWidth * oHeight + y * oWidth + x]);
          if (h_x[b * oWidth * oHeight + y * oWidth + x] !=
              h_x2[b * oWidth * oHeight + y * oWidth + x])
            allOk = false;
        }
        if (!allOk)
          printf(" FAILED CHECK ");
        printf("\n");
      }
      printf("------------------------\n");
    }
    printf("====================================\n");
  }

  return 0;
}

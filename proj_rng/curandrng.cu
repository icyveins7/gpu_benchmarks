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

    for (int i = 0; i < repeats; ++i) {
      rng.rand(d_x, oWidth, oHeight);

      thrust::host_vector<float> h_x = d_x;

      for (int b = 0; b < batch; ++b) {
        for (int y = 0; y < oHeight; ++y) {
          for (int x = 0; x < oWidth; ++x) {
            printf("%.3f ", h_x[b * oWidth * oHeight + y * oWidth + x]);
          }
          printf("\n");
        }
        printf("------------------------\n");
      }
      printf("====================================\n");
    }
  }

  return 0;
}

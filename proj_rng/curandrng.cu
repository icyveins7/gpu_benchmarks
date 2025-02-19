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

  return 0;
}

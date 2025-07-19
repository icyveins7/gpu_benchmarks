#include "thrust_extensions.h"
#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

int main(int argc, char *argv[]) {
  printf("Thrust RNGs\n");

  size_t numPts = 1000000;
  if (argc >= 2)
    numPts = strtol(argv[1], nullptr, 10);
  printf("Using numPts = %ld\n", numPts);

  int repeats = 5;
  if (argc >= 3)
    repeats = strtol(argv[2], nullptr, 10);
  printf("Using repeats = %d\n", repeats);

  // Test 2D RNG from example
  {
    printf("2D RNG with default_random_engine\n");
    thrust::device_vector<float> d_x(numPts);
    thrust::device_vector<float> d_y(numPts);
    Random_generator2d<thrust::random::default_random_engine> rnd;
    for (int i = 0; i < repeats; ++i) {
      thrust::generate(
          thrust::make_zip_iterator(
              thrust::make_tuple(d_x.begin(), d_y.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end())),
          rnd);
    }
    thrust::host_vector<float> h_x = d_x;
    thrust::host_vector<float> h_y = d_y;

    for (int i = 0; i < 10; ++i) {
      printf("%.3f, %.3f\n", h_x[i], h_y[i]);
    }
  }
  printf("====================================\n");
  // Test 2D RNG from example
  {
    printf("2D RNG with ranlux24\n");
    thrust::device_vector<float> d_x(numPts);
    thrust::device_vector<float> d_y(numPts);
    Random_generator2d<thrust::random::ranlux24> rnd;
    for (int i = 0; i < repeats; ++i) {
      thrust::generate(
          thrust::make_zip_iterator(
              thrust::make_tuple(d_x.begin(), d_y.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end())),
          rnd);
    }
    thrust::host_vector<float> h_x = d_x;
    thrust::host_vector<float> h_y = d_y;

    for (int i = 0; i < 10; ++i) {
      printf("%.3f, %.3f\n", h_x[i], h_y[i]);
    }
  }
  printf("====================================\n");

  // Test 1D RNG adapted from example
  {
    printf("1D RNG\n");
    thrust::device_vector<float> d_x(numPts);
    Random_generator1d rnd;

    for (int i = 0; i < repeats; ++i) {
      thrust::generate(d_x.begin(), d_x.end(), rnd);
    }
    thrust::host_vector<float> h_x = d_x;

    for (int i = 0; i < 10; ++i) {
      printf("%.3f\n", h_x[i]);
    }
  }
  printf("====================================\n");

  // Test 1D RNG with sincos to make complex
  {
    printf("1D RNG with sincos\n");
    Random_generator1d_phase rnd;

    thrust::device_vector<thrust::complex<float>> d_z(numPts);

    for (int i = 0; i < repeats; ++i) {
      thrust::generate(d_z.begin(), d_z.end(), rnd);
    }

    thrust::host_vector<thrust::complex<float>> h_z = d_z;

    for (int i = 0; i < 10; ++i) {
      printf("%.3f, %.3f\n", h_z[i].real(), h_z[i].imag());
    }
  }
  printf("====================================\n");

  // Test 1D RNG with sincos to make complex, then multiply with something
  {
    printf("1D RNG with sincos, then mul\n");
    Random_generator1d_phase_cplxMul rnd;

    thrust::device_vector<thrust::complex<float>> d_z(numPts);

    // Generate some numbers to multiply by
    thrust::device_vector<thrust::complex<float>> d_w(numPts);
    thrust::sequence(d_w.begin(), d_w.end());
    thrust::host_vector<thrust::complex<float>> h_w = d_w;

    for (int i = 0; i < repeats; ++i) {
      thrust::transform(d_w.begin(), d_w.end(), d_z.begin(), rnd);
    }

    thrust::host_vector<thrust::complex<float>> h_z = d_z;

    for (int i = 0; i < 10; ++i) {
      printf("%.3f, %.3f <- %.3f, %.3f\n", h_z[i].real(), h_z[i].imag(),
             h_w[i].real(), h_w[i].imag());
    }
  }
  printf("====================================\n");

  return 0;
}

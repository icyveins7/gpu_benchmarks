#include "kernels.h"
#include <gtest/gtest.h>
#include <random>

template <typename T, template <class> class U> void test_gridPolynom() {
  constexpr size_t length =
      12345; // use awkward number to check kernel correctness
  constexpr size_t numCoeffs = 5;

  // Instantiate the class template used
  U<T> gp(numCoeffs);

  // Generate random data on host
  thrust::host_vector<T> h_in(length);
  thrust::host_vector<T> h_out(length);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(0.0, 1.0);
  for (T &in : h_in) {
    in = dis(gen);
  }

  // Prepare device vectors
  thrust::device_vector<T> d_in(length);
  thrust::device_vector<T> d_out(length);
  d_in = h_in;

  // Run kernel and copy out to host
  gp.d_run(d_in, d_out);
  thrust::device_vector<T> d2h_out(length);
  d2h_out = d_out;

  // Run on host as well
  gp.h_run(h_in, h_out);

  // Check all output elements
  for (size_t i = 0; i < length; ++i) {
    EXPECT_FLOAT_EQ(h_out[i], d2h_out[i]);
  }
}

// Run the tests on all classes

TEST(NaiveGridPolynom, float) { test_gridPolynom<float, NaiveGridPolynom>(); }
TEST(NaiveGridPolynom, double) { test_gridPolynom<double, NaiveGridPolynom>(); }

TEST(SharedCoeffGridPolynom, float) {
  test_gridPolynom<float, SharedCoeffGridPolynom>();
}
TEST(SharedCoeffGridPolynom, double) {
  test_gridPolynom<double, SharedCoeffGridPolynom>();
}

TEST(ConstantCoeffGridPolynom, float) {
  test_gridPolynom<float, ConstantCoeffGridPolynom>();
}
TEST(ConstantCoeffGridPolynom, double) {
  test_gridPolynom<double, ConstantCoeffGridPolynom>();
}

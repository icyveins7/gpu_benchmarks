#include "transpose.cuh"

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

TEST(Transpose, RectangularMatrix) {
  constexpr int rows = 2;
  constexpr int cols = 3;
  thrust::host_vector<int> h_in{1, 2, 3, 4, 5, 6};
  thrust::device_vector<int> d_in = h_in;
  thrust::device_vector<int> d_out(rows * cols);

  transpose<int>(d_out.data().get(), d_in.data().get(), rows, cols);

  thrust::host_vector<int> h_out = d_out;
  const int expected[] = {1, 4, 2, 5, 3, 6};

  for (int i = 0; i < rows * cols; ++i)
    ASSERT_EQ(h_out[i], expected[i]);
}

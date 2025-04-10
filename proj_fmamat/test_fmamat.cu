#include "../include/pinnedalloc.cuh"
#include "fmamat.h"
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

template <typename Tcalc, typename Tx, typename Tm, typename Tc, typename Ty>
void test_naive_fmamat(int width, int height) {
  thrust::pinned_host_vector<Tx> src(width * height);
  thrust::pinned_host_vector<Ty> dst(width * height);
  thrust::pinned_host_vector<float> m(width);
  thrust::pinned_host_vector<float> c(width);

  // Some initial values
  thrust::sequence(src.begin(), src.end(), 1.0f);
  thrust::sequence(m.begin(), m.end(), 1.0f);
  thrust::sequence(c.begin(), c.end(), 0.1f, 0.1f);

  thrust::device_vector<Tx> d_src = src;
  thrust::device_vector<Ty> d_dst = dst;
  thrust::device_vector<float> d_m = m;
  thrust::device_vector<float> d_c = c;

  dim3 threads_per_blk(128);
  dim3 blks_per_grid(d_src.size() / threads_per_blk.x + 1);
  naive_fmamat_columns_kernel<Tcalc><<<blks_per_grid, threads_per_blk>>>(
      d_src.data().get(), height, width, d_m.data().get(), d_c.data().get(),
      d_dst.data().get());

  dst = d_dst;

  // Generate CPU validated checks
  thrust::pinned_host_vector<Ty> check(dst.size());
  validate_fmamat_columns<float>(src, height, width, m, c, check);

  // Check output
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      EXPECT_FLOAT_EQ((float)dst[i * width + j], (float)check[i * width + j]);
    }
  }
}

// Run the tests for different template combinations and dimensions
// Format is Tcalc_TxTmTc_Ty

TEST(NaiveFmaMatColumns, float_floatfloatfloat_float) {
  test_naive_fmamat<float, float, float, float, float>(8, 8);
  // Some odd numbers
  test_naive_fmamat<float, float, float, float, float>(17, 17);
  // Some large numbers
  test_naive_fmamat<float, float, float, float, float>(1024, 1024);
  // Rectangles
  test_naive_fmamat<float, float, float, float, float>(65, 13);
}

#include "accessors.cuh"

#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>
#include <thrust/sequence.h>

template <typename T>
__global__ void test_roiLoad_kernel(const T *src, T *dst, const int srcWidth,
                                    const int srcHeight, const int roiStartX,
                                    const int roiStartY, const int roiLengthX,
                                    const int roiLengthY) {
  // Load and copy directly
  gridRoiLoad<T>(src, srcWidth, srcHeight, roiStartX, roiLengthX, roiStartY,
                 roiLengthY, dst);
}

template <typename T>
void test_roiLoad(const int srcWidth, const int srcHeight, const int roiStartX,
                  const int roiStartY, const int roiLengthX,
                  const int roiLengthY) {
  thrust::device_vector<T> d_src(srcWidth * srcHeight);
  thrust::sequence(d_src.begin(), d_src.end());
  thrust::device_vector<T> d_dst(roiLengthX * roiLengthY);

  test_roiLoad_kernel<<<4, 32>>>(d_src.data().get(), d_dst.data().get(),
                                 srcWidth, srcHeight, roiStartX, roiStartY,
                                 roiLengthX, roiLengthY);

  thrust::host_vector<T> h_dst = d_dst;

  for (int i = 0; i < roiLengthY; ++i) {
    for (int j = 0; j < roiLengthX; ++j) {
      EXPECT_EQ(h_dst[i * roiLengthX + j],
                (i + roiStartY) * srcWidth + j + roiStartX);
    }
  }
}

TEST(Accessors, roiLoad) { test_roiLoad<unsigned short>(9, 4, 3, 1, 3, 2); }

#include "kernels.h"
#include <gtest/gtest.h>
#include <random>
#include <npp.h>

template <typename T, template <class> class U>
void test_remap(const size_t height, const size_t width, const size_t srcHeight,
                const size_t srcWidth) {
  // Instantiate the class template used
  // First we generate simple sequence of values
  U<T> remap(srcHeight, srcWidth);

  // Make some input and output vectors
  thrust::host_vector<float> h_inX(width * height);
  thrust::host_vector<float> h_inY(width * height);
  thrust::host_vector<T> h_out(width * height);

  // Generate some input
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (float &in : h_inX) {
    in = dis(gen) * srcWidth;
  }
  for (float &in : h_inY) {
    in = dis(gen) * srcHeight;
  }
  // We specifically go out of bounds on the last few elements
  h_inX[width * height - 1] = srcWidth;
  h_inY[width * height - 1] = 0;

  h_inX[width * height - 2] = 0;
  h_inY[width * height - 2] = srcHeight;

  // Copy to device vectors
  thrust::device_vector<float> d_inX(width * height);
  thrust::device_vector<float> d_inY(width * height);
  thrust::device_vector<T> d_out(width * height);
  d_inX = h_inX;
  d_inY = h_inY;

  // Pre-set d_out to be all max value
  thrust::fill(d_out.begin(), d_out.end(), std::numeric_limits<T>::max());

  // Run kernel
  remap.d_run(d_inX, d_inY, width, height, d_out);
  thrust::host_vector<T> d2h_out(width * height);
  d2h_out = d_out;

  // Run CPU-side test
  // Pre-set h_out to be all max value
  thrust::fill(h_out.begin(), h_out.end(), std::numeric_limits<T>::max());
  remap.h_run(h_inX, h_inY, width, height, h_out);

  // Check all output elements
  for (size_t i = 0; i < width * height; ++i) {
    printf("Index %zd:\n", i);
    printf("Requested pixel coords are %.1f, %.1f\n", h_inX[i], h_inY[i]);
    printf("Top left  pixel is %.1f\n", (float)remap.get_h_src()[(size_t)floorf(h_inY[i])*srcWidth + (size_t)floorf(h_inX[i])]);
    printf("Top right pixel is %.1f\n", (float)remap.get_h_src()[(size_t)floorf(h_inY[i])*srcWidth + (size_t)floorf(h_inX[i]) + 1]);
    EXPECT_FLOAT_EQ(h_out[i], d2h_out[i]);
  }
}

// template <typename T, template <class> class U, typename F>
// void test_remap_vs_npp(const size_t height, const size_t width, const size_t srcHeight,
//                        const size_t srcWidth, F f) {
//   // Basically the same as above, but check against NPP implementation instead
//
//   // Instantiate the class template used
//   // First we generate simple sequence of values
//   U<T> remap(srcHeight, srcWidth);
//
//   // Make some input and output vectors
//   thrust::host_vector<float> h_inX(width * height);
//   thrust::host_vector<float> h_inY(width * height);
//
//   // Generate some input
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<float> dis(0.0, 1.0);
//   for (float &in : h_inX) {
//     in = dis(gen) * srcWidth;
//   }
//   for (float &in : h_inY) {
//     in = dis(gen) * srcHeight;
//   }
//   // We specifically go out of bounds on the last few elements
//   h_inX[width * height - 1] = srcWidth - 1 + 0.8;
//   h_inY[width * height - 1] = 0;
//
//   h_inX[width * height - 2] = 0;
//   h_inY[width * height - 2] = srcHeight - 1 + 0.8;
//
//   // Still within 1 pixel out
//   h_inX[width * height - 3] = 0;
//   h_inY[width * height - 3] = -0.8;
//
//   h_inX[width * height - 4] = -0.8;
//   h_inY[width * height - 4] = 0;
//
//   // Now 2 pixels out
//   h_inX[width * height - 5] = 0;
//   h_inY[width * height - 5] = -1.8;
//
//   h_inX[width * height - 6] = -1.8;
//   h_inY[width * height - 6] = 0;
//
//   h_inX[width * height - 7] = srcWidth + 0.8;
//   h_inY[width * height - 7] = 0;
//
//   h_inX[width * height - 8] = 0;
//   h_inY[width * height - 8] = srcHeight + 0.8;
//
//   // Copy to device vectors
//   thrust::device_vector<float> d_inX(width * height);
//   thrust::device_vector<float> d_inY(width * height);
//   thrust::device_vector<T> d_out(width * height);
//   d_inX = h_inX;
//   d_inY = h_inY;
//
//   // Pre-set d_out to be all max value
//   thrust::fill(d_out.begin(), d_out.end(), std::numeric_limits<T>::max());
//
//   // Run kernel
//   remap.d_run(d_inX, d_inY, width, height, d_out);
//   thrust::host_vector<T> d2h_out(width * height);
//   d2h_out = d_out;
//
//   // Run NPP test
//   // Pre-set h_out to be all max value
//   thrust::device_vector<T> d_nppout(width * height);
//   thrust::fill(d_nppout.begin(), d_nppout.end(), std::numeric_limits<T>::max());
//
//   NppiSize srcSize = {(int)srcWidth, (int)srcHeight};
//   NppiRect srcROI = {0, 0, (int)srcWidth, (int)srcHeight};
//   NppiSize dstSize = {(int)width, (int)height};
//   NppStatus status = f(
//     thrust::raw_pointer_cast(remap.get_d_src().data()),
//     srcSize,
//     srcWidth * sizeof(T),
//     srcROI,
//     thrust::raw_pointer_cast(d_inX.data()),
//     width * sizeof(float),
//     thrust::raw_pointer_cast(d_inY.data()),
//     width * sizeof(float),
//     thrust::raw_pointer_cast(d_nppout.data()),
//     width * sizeof(T),
//     dstSize,
//     NPPI_INTER_LINEAR
//   );
//   // Copy back to host
//   thrust::host_vector<T> h_nppout = d_nppout;
//
//   // Check all output elements
//   for (size_t i = 0; i < width * height; ++i) {
//     printf("Index %zd:\n", i);
//     printf("Requested pixel coords are %.1f, %.1f\n", h_inX[i], h_inY[i]);
//     // printf("Top left  pixel is %.1f\n", (float)remap.get_h_src()[(size_t)floorf(h_inY[i])*srcWidth + (size_t)floorf(h_inX[i])]);
//     // printf("Top right pixel is %.1f\n", (float)remap.get_h_src()[(size_t)floorf(h_inY[i])*srcWidth + (size_t)floorf(h_inX[i]) + 1]);
//     EXPECT_FLOAT_EQ(h_nppout[i], d2h_out[i]);
//   }
// }



// Run the tests on all classes

// =========== Test on floats
TEST(NaiveRemap, float_128x128_128x128){
  test_remap<float, NaiveRemap>(128, 128, 128, 128);
  // test_remap_vs_npp<float, NaiveRemap>(128, 128, 128, 128, nppiRemap_32f_C1R);
}
TEST(NaiveRemap, float_32x32_128x128){
  test_remap<float, NaiveRemap>(32, 32, 128, 128);
}
TEST(NaiveRemap, float_32x32_123x123){
  test_remap<float, NaiveRemap>(32, 32, 123, 123);
}
TEST(NaiveRemap, float_25x25_123x123){
  test_remap<float, NaiveRemap>(25, 25, 123, 123);
}


// =========== Test on uint16_t
TEST(NaiveRemap, uint16_t_128x128_128x128) {
  test_remap<uint16_t, NaiveRemap>(128, 128, 128, 128);
}
TEST(NaiveRemap, uint16_t_32x32_128x128) {
  test_remap<uint16_t, NaiveRemap>(32, 32, 128, 128);
}
TEST(NaiveRemap, uint16_t_32x32_123x123) {
  test_remap<uint16_t, NaiveRemap>(32, 32, 123, 123);
}
TEST(NaiveRemap, uint16_t_25x25_123x123) {
  test_remap<uint16_t, NaiveRemap>(25, 25, 123, 123);
}

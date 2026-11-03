#include "kernels.h"
#include "overlapKernels.cuh"
#include <gtest/gtest.h>
#include <limits>
#include <npp.h>
#include <random>
#include <thrust/execution_policy.h>
#include <vector>

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
  h_inX[width * height - 1] = srcWidth + 2.0f;
  h_inY[width * height - 1] = 0;

  h_inX[width * height - 2] = 0;
  h_inY[width * height - 2] = srcHeight + 2.0f;

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
    // printf("Index %zd:\n", i);
    // printf("Requested pixel coords are %.1f, %.1f\n", h_inX[i], h_inY[i]);
    // printf("Top left  pixel is %.1f\n",
    //        (float)remap.get_h_src()[(size_t)floorf(h_inY[i]) * srcWidth +
    //                                 (size_t)floorf(h_inX[i])]);
    // printf("Top right pixel is %.1f\n",
    //        (float)remap.get_h_src()[(size_t)floorf(h_inY[i]) * srcWidth +
    //                                 (size_t)floorf(h_inX[i]) + 1]);
    EXPECT_FLOAT_EQ(h_out[i], d2h_out[i]);
  }
}

template <typename T, template <class> class U> void test_remap_edgecases() {
  // We make a very simple 2x2 source
  thrust::host_vector<T> h_src(4);
  h_src[0] = 1.0;
  h_src[1] = 2.0;
  h_src[2] = 3.0;
  h_src[3] = 4.0;
  U<T> remap(h_src, 2, 2);

  /*
  For the destination we specifically create a 5x5 grid, where 4 of the points
  coincide with the original 4 pixels, and 4 other points are midpoint along the
  edges.

  X X X X X
  X @ . @ X
  X . . . X
  X @ . @ X
  X X X X X

  X : points outside the src, expect no output
  @ : points coinciding with src pixels, expect same value
  . : points on the edge of or inside the src, expect valid interpolation

  */

  thrust::device_vector<T> d_dest(25);
  thrust::fill(d_dest.begin(), d_dest.end(), std::numeric_limits<T>::max());

  // Write the exact requested pixel coordinates so we can be sure
  thrust::host_vector<float> h_inX(25);
  thrust::host_vector<float> h_inY(25);
  // clang-format off
  h_inX[0] = -0.5f; h_inX[1] = 0.0f; h_inX[2] = 0.5f; h_inX[3] = 1.0f; h_inX[4] = 1.5f;
  h_inX[5] = -0.5f; h_inX[6] = 0.0f; h_inX[7] = 0.5f; h_inX[8] = 1.0f; h_inX[9] = 1.5f;
  h_inX[10] = -0.5f; h_inX[11] = 0.0f; h_inX[12] = 0.5f; h_inX[13] = 1.0f; h_inX[14] = 1.5f;
  h_inX[15] = -0.5f; h_inX[16] = 0.0f; h_inX[17] = 0.5f; h_inX[18] = 1.0f; h_inX[19] = 1.5f;
  h_inX[20] = -0.5f; h_inX[21] = 0.0f; h_inX[22] = 0.5f; h_inX[23] = 1.0f; h_inX[24] = 1.5f;


  h_inY[0] = -0.5f; h_inY[1] = -0.5f; h_inY[2] = -0.5f; h_inY[3] = -0.5f; h_inY[4] = -0.5f;
  h_inY[5] = 0.0f; h_inY[6] = 0.0f; h_inY[7] = 0.0f; h_inY[8] = 0.0f; h_inY[9] = 0.0f;
  h_inY[10] = 0.5f; h_inY[11] = 0.5f; h_inY[12] = 0.5f; h_inY[13] = 0.5f; h_inY[14] = 0.5f;
  h_inY[15] = 1.0f; h_inY[16] = 1.0f; h_inY[17] = 1.0f; h_inY[18] = 1.0f; h_inY[19] = 1.0f;
  h_inY[20] = 1.5f; h_inY[21] = 1.5f; h_inY[22] = 1.5f; h_inY[23] = 1.5f; h_inY[24] = 1.5f;

  thrust::device_vector<float> d_inX(25);
  thrust::device_vector<float> d_inY(25);
  d_inX = h_inX;
  d_inY = h_inY;

  // Run the remap
  remap.d_run(d_inX, d_inY, 5, 5, d_dest);
  thrust::host_vector<T> h_dest(25);
  h_dest = d_dest;

  // Run explicit expectations
  constexpr T invalid = std::numeric_limits<T>::max();

  thrust::host_vector<float> h_src_float(h_src.size());
  for (size_t i = 0; i < h_src.size(); ++i) {
    h_src_float[i] = static_cast<float>(h_src[i]);
  }

  // 1st row
  EXPECT_FLOAT_EQ(h_dest[0], invalid);
  EXPECT_FLOAT_EQ(h_dest[1], invalid);
  EXPECT_FLOAT_EQ(h_dest[2], invalid);
  EXPECT_FLOAT_EQ(h_dest[3], invalid);
  EXPECT_FLOAT_EQ(h_dest[4], invalid);

  // 2nd row
  EXPECT_FLOAT_EQ(h_dest[5], invalid);
  EXPECT_FLOAT_EQ(h_dest[6], static_cast<T>(h_src_float[0]));                  // top left point
  EXPECT_FLOAT_EQ(h_dest[7], static_cast<T>((h_src_float[0] + h_src_float[1]) / 2)); // top edge mid point
  EXPECT_FLOAT_EQ(h_dest[8], static_cast<T>(h_src_float[1]));                  // top right point
  EXPECT_FLOAT_EQ(h_dest[9], invalid);

  // 3rd row
  EXPECT_FLOAT_EQ(h_dest[10], invalid);
  EXPECT_FLOAT_EQ(h_dest[11], static_cast<T>((h_src_float[0] + h_src_float[2]) / 2)); // left edge mid point
  EXPECT_FLOAT_EQ(h_dest[12], static_cast<T>((h_src_float[0] + h_src_float[1] + h_src_float[2] + h_src_float[3]) / 4)); // middle of square
  EXPECT_FLOAT_EQ(h_dest[13], static_cast<T>((h_src_float[1] + h_src_float[3]) / 2)); // right edge mid point
  EXPECT_FLOAT_EQ(h_dest[14], invalid);

  // 4th row
  EXPECT_FLOAT_EQ(h_dest[15], invalid);
  EXPECT_FLOAT_EQ(h_dest[16], static_cast<T>(h_src_float[2]));                  // btm left point
  EXPECT_FLOAT_EQ(h_dest[17], static_cast<T>((h_src_float[2] + h_src_float[3]) / 2)); // btm edge mid point
  EXPECT_FLOAT_EQ(h_dest[18], static_cast<T>(h_src_float[3]));                  // btm right point
  EXPECT_FLOAT_EQ(h_dest[19], invalid);

  // 5th row
  EXPECT_FLOAT_EQ(h_dest[20], invalid);
  EXPECT_FLOAT_EQ(h_dest[21], invalid);
  EXPECT_FLOAT_EQ(h_dest[22], invalid);
  EXPECT_FLOAT_EQ(h_dest[23], invalid);
  EXPECT_FLOAT_EQ(h_dest[24], invalid);

  // clang-format on
}

template <typename T, template <class> class U>
void test_remap_wraparound_edgecases() {

  // We now make a very 3x2 source so we can move 'out of bounds'
  const size_t srcHeight = 3;
  const size_t srcWidth = 2;
  thrust::host_vector<T> h_src(srcHeight * srcWidth);
  // Set values 1 to 6 i.e.
  // 1 2
  // 3 4
  // 5 6
  thrust::sequence(thrust::host, h_src.begin(), h_src.end(), 1.0f);
  U<T> remap(h_src, srcHeight, srcWidth);

  /*
  For the destination we now create a 7x7 grid, where the original 5x5 points
  from the naive case are repeated, but now we add 1 more column to test the
  range to the right, which is expected to work now.

  X X X X X ^ ^
  X @ . @ ^ % ^
  X . . . ^ ^ ^
  ^ @ . @ ^ % $
  ^ . . . X X X
  ^ @ . @ X X X
  X X X X X X X

  X : points outside the allowed bounds, expect no output
  @ : points coinciding with src pixels, expect same value
  . : points on the edge of or inside the src, expect valid interpolation
  ^ : points utilising the wraparound, which should have output
  % : points in wraparound that coincide with a src pixel in subsequent rows
  $ : special case to test where we specifically extend completely beyond the
      input, so expect no output

  */
  // clang-format off

  // I am going to use std::vector to create the values and then copy
  // since this lets me use the less verbose (and I would think) clearer
  // list initialization (which thrust vectors do not support)
  std::vector<float> tmpX{
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 100.0f, // this point is to test far out of bounds wrap
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
    -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f
  };
  std::vector<float> tmpY{
    -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
     0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.5f,  1.5f,  1.5f,  1.5f,  1.5f,  1.5f,  1.5f,
     2.0f,  2.0f,  2.0f,  2.0f,  2.0f,  2.0f,  2.0f,
     2.5f,  2.5f,  2.5f,  2.5f,  2.5f,  2.5f,  2.5f
  };
  thrust::host_vector<float> h_inX(tmpX.begin(), tmpX.end());
  thrust::host_vector<float> h_inY(tmpY.begin(), tmpY.end());
  thrust::device_vector<float> d_inX = h_inX;
  thrust::device_vector<float> d_inY = h_inY;

  // Run the remap
  thrust::device_vector<T> d_dest(h_inX.size());
  // same as before, fix value to max so we can see if no output at a pixel
  thrust::fill(d_dest.begin(), d_dest.end(), std::numeric_limits<T>::max());
  remap.d_run(d_inX, d_inY, 7, 7, d_dest);
  thrust::host_vector<T> h_dest = d_dest;

  // Run explicit expectations
  // Define the invalid value again
  constexpr T inv = std::numeric_limits<T>::max();
  // Define expectations as floating values
  // Let's be even more explicit here and just write the value
  std::vector<float> h_expect{
    inv,  inv,  inv,  inv,  inv, 2.0f, 2.5f,
    inv, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
    inv, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f,
   2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f,  inv,
   3.5f, 4.0f, 4.5f, 5.0f,  inv,  inv,  inv,
   4.5f, 5.0f, 5.5f, 6.0f,  inv,  inv,  inv,
    inv,  inv,  inv,  inv,  inv,  inv,  inv
  };

  for (size_t i = 0; i < h_expect.size(); ++i){
    printf("x/y = %.1f, %.1f\n", h_inX[i], h_inY[i]);
    printf("expect = %.1f, result = %.1f\n", (float)static_cast<T>(h_expect[i]), (float)h_dest[i]);
    EXPECT_FLOAT_EQ(h_dest[i], static_cast<T>(h_expect[i]));
  }

  // clang-format on
}

// template <typename T, template <class> class U, typename F>
// void test_remap_vs_npp(const size_t height, const size_t width, const size_t
// srcHeight,
//                        const size_t srcWidth, F f) {
//   // Basically the same as above, but check against NPP implementation
//   instead
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
//   thrust::fill(d_nppout.begin(), d_nppout.end(),
//   std::numeric_limits<T>::max());
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
//     // printf("Top left  pixel is %.1f\n",
//     (float)remap.get_h_src()[(size_t)floorf(h_inY[i])*srcWidth +
//     (size_t)floorf(h_inX[i])]);
//     // printf("Top right pixel is %.1f\n",
//     (float)remap.get_h_src()[(size_t)floorf(h_inY[i])*srcWidth +
//     (size_t)floorf(h_inX[i]) + 1]); EXPECT_FLOAT_EQ(h_nppout[i], d2h_out[i]);
//   }
// }

// Run the tests on all classes

// =========== Test NaiveRemap on floats
TEST(NaiveRemap, float_128x128_128x128) {
  test_remap<float, NaiveRemap>(128, 128, 128, 128);
  // test_remap_vs_npp<float, NaiveRemap>(128, 128, 128, 128,
  // nppiRemap_32f_C1R);
}
TEST(NaiveRemap, float_32x32_128x128) {
  test_remap<float, NaiveRemap>(32, 32, 128, 128);
}
TEST(NaiveRemap, float_32x32_123x123) {
  test_remap<float, NaiveRemap>(32, 32, 123, 123);
}
TEST(NaiveRemap, float_25x25_123x123) {
  test_remap<float, NaiveRemap>(25, 25, 123, 123);
}

TEST(NaiveRemap, float_edgecase) { test_remap_edgecases<float, NaiveRemap>(); }

// =========== Test NaiveRemap on uint16_t
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
TEST(NaiveRemap, uint16_t_edgecase) {
  test_remap_edgecases<uint16_t, NaiveRemap>();
}
// =========== Test NaiveRemapWraparound on floats
TEST(NaiveRemapWraparound, float_128x128_128x128) {
  test_remap<float, NaiveRemapWraparound>(128, 128, 128, 128);
}
TEST(NaiveRemapWraparound, float_32x32_128x128) {
  test_remap<float, NaiveRemapWraparound>(32, 32, 128, 128);
}
TEST(NaiveRemapWraparound, float_32x32_123x123) {
  test_remap<float, NaiveRemapWraparound>(32, 32, 123, 123);
}
TEST(NaiveRemapWraparound, float_25x25_123x123) {
  test_remap<float, NaiveRemapWraparound>(25, 25, 123, 123);
}

TEST(NaiveRemapWraparound, float_edgecase) {
  test_remap_wraparound_edgecases<float, NaiveRemapWraparound>();
}

// =========== Test NaiveRemapWraparound on uint16_ts
TEST(NaiveRemapWraparound, uint16_t_128x128_128x128) {
  test_remap<uint16_t, NaiveRemapWraparound>(128, 128, 128, 128);
}
TEST(NaiveRemapWraparound, uint16_t_32x32_128x128) {
  test_remap<uint16_t, NaiveRemapWraparound>(32, 32, 128, 128);
}
TEST(NaiveRemapWraparound, uint16_t_32x32_123x123) {
  test_remap<uint16_t, NaiveRemapWraparound>(32, 32, 123, 123);
}
TEST(NaiveRemapWraparound, uint16_t_25x25_123x123) {
  test_remap<uint16_t, NaiveRemapWraparound>(25, 25, 123, 123);
}
TEST(NaiveRemapWraparound, uint16_t_edgecase) {
  test_remap_wraparound_edgecases<uint16_t, NaiveRemapWraparound>();
}

/**
 * @brief Four complete input pixels contribute equally to one output pixel.
 *
 * A 2x2 input containing 10, 20, 30, 40 lies entirely inside one output
 * pixel twice as wide, so each contributes weight 0.25 and the result is 25.
 * Invalidating 40 leaves weight 0.75 and mean 20: this passes 0.45 but fails
 * a threshold of 0.75 because the validity comparison is strictly greater.
 */
TEST(OverlapRemap, AreaWeightsAndStrictThreshold) {
  containers::DeviceImageStorage<float> input(2, 2);
  thrust::host_vector<float> values(4);
  values[0] = 10;
  values[1] = 20;
  values[2] = 30;
  values[3] = 40;
  input.vec = values;
  containers::DeviceImageStorage<float> output(1, 1);
  const overlap::OutputToInput<float> transform{{2, 0}, {0, 2}, {0.5, 0.5}};
  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<float> result = output.vec;
  EXPECT_FLOAT_EQ(result[0], 25.0f);

  values[3] = std::numeric_limits<float>::max();
  input.vec = values;
  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  result = output.vec;
  EXPECT_FLOAT_EQ(result[0], 20.0f);

  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.75f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  result = output.vec;
  EXPECT_EQ(result[0], std::numeric_limits<float>::max());
}

/**
 * @brief A 45-degree output square overlaps four equal input quadrants.
 *
 * The first output pixel is centred at (0.5, 0.5) and rotated 45 degrees.
 * It covers equal areas of input values 10, 20, 30, 40, giving mean 25 and
 * total weight 1. At threshold 0.95 it is valid; at 1.01 it is invalid.
 * Other output columns have insufficient overlap (or none) and are invalid.
 */
TEST(OverlapRemap, RotatedOutput) {
  containers::DeviceImageStorage<float> input(2, 2);
  thrust::host_vector<float> values(4);
  values[0] = 10;
  values[1] = 20;
  values[2] = 30;
  values[3] = 40;
  input.vec = values;
  containers::DeviceImageStorage<float> output(35, 1);
  const double c = std::sqrt(0.5);
  const overlap::OutputToInput<double> transform{
      {c, c}, {-c, c}, {0.5, 0.5}};
  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.95,
                 overlap::AreaAccumulator<double>{}, dim3(1, 1), dim3(8, 2));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<float> result = output.vec;
  EXPECT_NEAR(result[0], 25.0f, 1e-5f);
  for (int i = 1; i < output.width; ++i)
    EXPECT_EQ(result[i], std::numeric_limits<float>::max());

  overlap::remapOverlap(input.cimage(), output.image(), transform, 1.01,
                 overlap::AreaAccumulator<double>{}, dim3(1, 1), dim3(8, 2));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  result = output.vec;
  EXPECT_EQ(result[0], std::numeric_limits<float>::max());
}

/**
 * @brief An identity remap must fill every pixel with either launch shape.
 *
 * A 35x9 input and output share pixel centres, so every output must exactly
 * match its corresponding input value. The default 32x4 block gets an
 * automatically sized 2x3 grid. A single 8x2 block then has to stride in
 * both X and Y to produce the same complete 35x9 result.
 */
TEST(OverlapRemap, GridStrideBothAxes) {
  containers::DeviceImageStorage<float> input(35, 9);
  thrust::host_vector<float> values(input.vec.size());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i + 1);
  input.vec = values;
  containers::DeviceImageStorage<float> output(35, 9);
  const overlap::OutputToInput<double> transform{{1, 0}, {0, 1}, {0, 0}};
  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<float> result = output.vec;
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_FLOAT_EQ(result[i], values[i]);

  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45,
                 overlap::AreaAccumulator<double>{}, dim3(1, 1), dim3(8, 2));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  result = output.vec;
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_FLOAT_EQ(result[i], values[i]);
}

/** @brief Test policy that weights the square of each input value by area. */
struct SquaredValueAccumulator {
  __device__ overlap::Contribution<double>
  operator()(const cuda_vec2_t<double> (&quad)[4], int x, int y,
             float sample, double areaScale) const {
    const double weight = overlap::intersectionArea<double>(quad, x, y) *
                          areaScale;
    return {weight, weight * sample * sample};
  }
};

/**
 * @brief Partially covered input pixels use their intersecting areas.
 *
 * The output square centred at (0.25, 0.25) overlaps the 2x2 inputs by
 * 9/16, 3/16, 3/16, and 1/16 respectively. With values 10, 20, 30, 40,
 * ordinary area weighting gives 17.5. A replacement accumulator applying
 * those same weights to squared values instead gives 400.
 */
TEST(OverlapRemap, FractionalIntersectionAndCustomAccumulator) {
  containers::DeviceImageStorage<float> input(2, 2);
  thrust::host_vector<float> values(4);
  values[0] = 10;
  values[1] = 20;
  values[2] = 30;
  values[3] = 40;
  input.vec = values;
  containers::DeviceImageStorage<float> output(1, 1);
  const overlap::OutputToInput<double> transform{{1, 0}, {0, 1},
                                                  {0.25, 0.25}};
  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<float> result = output.vec;
  EXPECT_FLOAT_EQ(result[0], 17.5f);

  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45,
                 SquaredValueAccumulator{});
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  result = output.vec;
  EXPECT_FLOAT_EQ(result[0], 400.0f);
}

/**
 * @brief Compare a four-Z-worker launch against the one-thread-per-output path.
 *
 * The image contains distinct values and invalid pixels. A tilted output grid
 * has interior, partially covered, and fully outside pixels. One 32x2x4 block
 * must stride over both output axes and safely handle incomplete edge tiles.
 */
void checkSharedOverlap() {
  containers::DeviceImageStorage<float> input(32, 16);
  thrust::host_vector<float> values(input.vec.size());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i % 97 + 1);
  values[7] = std::numeric_limits<float>::max();
  values[36] = std::numeric_limits<float>::max();
  input.vec = values;
  containers::DeviceImageStorage<float> output(35, 9);
  const overlap::OutputToInput<double> transform{
      {1.1, 0.14}, {-0.3, 1.2}, {-0.7, 0.1}};

  overlap::remapOverlap(input.cimage(), output.image(), transform, 0.45);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const thrust::host_vector<float> expected = output.vec;

  overlap::remapOverlapSharedPartials(
      input.cimage(), output.image(), transform, 0.45,
      overlap::AreaAccumulator<double>{}, dim3(1, 1), dim3(32, 2, 4));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const thrust::host_vector<float> actual = output.vec;
  bool foundValid = false, foundInvalid = false;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (expected[i] == std::numeric_limits<float>::max()) {
      EXPECT_EQ(actual[i], expected[i]) << "pixel " << i;
      foundInvalid = true;
    } else {
      EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "pixel " << i;
      foundValid = true;
    }
  }
  EXPECT_TRUE(foundValid);
  EXPECT_TRUE(foundInvalid);

  containers::DeviceImageStorage<float> smallInput(2, 2);
  thrust::host_vector<float> smallValues(4, 1.0f);
  smallValues[3] = std::numeric_limits<float>::max();
  smallInput.vec = smallValues;
  containers::DeviceImageStorage<float> smallOutput(1, 1);
  const overlap::OutputToInput<float> scaled{{2, 0}, {0, 2}, {0.5, 0.5}};
  overlap::remapOverlapSharedPartials(smallInput.cimage(), smallOutput.image(),
                                      scaled, 0.75f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<float> strictResult = smallOutput.vec;
  EXPECT_EQ(strictResult[0], std::numeric_limits<float>::max());

  thrust::host_vector<float> fractionalValues(4);
  fractionalValues[0] = 10;
  fractionalValues[1] = 20;
  fractionalValues[2] = 30;
  fractionalValues[3] = 40;
  smallInput.vec = fractionalValues;
  const overlap::OutputToInput<double> fractional{{1, 0}, {0, 1},
                                                    {0.25, 0.25}};
  overlap::remapOverlapSharedPartials(smallInput.cimage(), smallOutput.image(),
                                      fractional, 0.45,
                                      SquaredValueAccumulator{});
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<float> squaredResult = smallOutput.vec;
  EXPECT_FLOAT_EQ(squaredResult[0], 400.0f);
}

/** @brief Shared per-worker partials must agree with the baseline. */
TEST(OverlapRemap, SharedPartialsMatchBaseline) {
  checkSharedOverlap();
}

#include "containers/cubwrappers.cuh"

#include "gtest/gtest.h"

// Tests for cubw::helpers functors — all __host__ __device__, exercised on
// host.

TEST(ContainersCubwHelpers, RowIndexIterator) {
  // width=3, 2 rows → 6 elements; expected keys: 0 0 0 | 1 1 1
  const int width = 3;
  auto iter = cubw::helpers::makeRowIndexIterator<int>(width);

  ASSERT_EQ(iter[0], 0);
  ASSERT_EQ(iter[1], 0);
  ASSERT_EQ(iter[2], 0);
  ASSERT_EQ(iter[3], 1);
  ASSERT_EQ(iter[4], 1);
  ASSERT_EQ(iter[5], 1);
}

TEST(ContainersCubwHelpers, RowStridedIterator) {
  // 4x3 image (M=4 rows, N=3 cols), row_stride=2 → access rows 0 and 2
  // row 0: [ 0,  1,  2 ] ← iter[0..2]
  // row 1: [ 3,  4,  5 ]
  // row 2: [ 6,  7,  8 ] ← iter[3..5]
  // row 3: [ 9, 10, 11 ]
  int image[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto iter = cubw::helpers::makeRowStridedIterator(image, 3, 2);

  ASSERT_EQ(iter[0], 0);
  ASSERT_EQ(iter[1], 1);
  ASSERT_EQ(iter[2], 2);
  ASSERT_EQ(iter[3], 6);
  ASSERT_EQ(iter[4], 7);
  ASSERT_EQ(iter[5], 8);
}

TEST(ContainersCubwHelpers, StridedRowIndexIterator) {
  // width=10, row_stride=4: selected rows map to actual rows 0, 4, 8, ...
  // selected row 0 (indices  0- 9) → flat offsets  0- 9
  // selected row 1 (indices 10-19) → flat offsets 40-49
  // selected row 2 (indices 20-29) → flat offsets 80-89
  const int width = 10, row_stride = 4;
  auto iter =
      cubw::helpers::makeRowStridedIndexIterator<int>(width, row_stride);

  ASSERT_EQ(iter[0], 0);
  ASSERT_EQ(iter[9], 9);
  ASSERT_EQ(iter[10], 40);
  ASSERT_EQ(iter[19], 49);
  ASSERT_EQ(iter[20], 80);
  ASSERT_EQ(iter[29], 89);
}

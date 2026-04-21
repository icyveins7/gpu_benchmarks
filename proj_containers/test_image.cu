#include "containers/image.cuh"
#include "gtest/gtest.h"
#include <numeric>

TEST(ContainersDeviceImageStorage, BasicChecks) {
  containers::DeviceImageStorage<float> img(10, 10);
  EXPECT_EQ(10, img.width);
  EXPECT_EQ(10, img.height);
  EXPECT_EQ(img.vec.size(), 10 * 10);
  EXPECT_EQ(img.vec.capacity(), 10 * 10);

  img.resize(20, 20);
  EXPECT_EQ(20, img.width);
  EXPECT_EQ(20, img.height);
  EXPECT_EQ(img.vec.size(), 20 * 20);
  EXPECT_EQ(img.vec.capacity(), 20 * 20);

  img.resize(10, 10);
  EXPECT_EQ(10, img.width);
  EXPECT_EQ(10, img.height);
  EXPECT_EQ(img.vec.size(), 10 * 10);
  EXPECT_EQ(img.vec.capacity(), 20 * 20);
}

TEST(ContainersImageTile, BasicChecks) {
  unsigned int width = 4, height = 4;
  std::vector<int> vec(width * height);
  containers::Image<int> img(vec.data(), width, height);
  std::iota(vec.begin(), vec.end(), 1);

  /*
  1  2  3  4
  5  6  7  8
  9  10 11 12
  13 14 15 16
  */

  // centre 2x2 tile
  unsigned int twidth = 2, theight = 2;
  unsigned int startRow = 1, startCol = 1;
  std::vector<int> vect(twidth * theight);
  containers::ImageTile<int> tile(vect.data(), twidth, theight, startRow,
                                  startCol);

  EXPECT_FALSE(tile.rowIsValid(0));
  EXPECT_FALSE(tile.colIsValid(0));
  EXPECT_TRUE(tile.rowIsValid(1));
  EXPECT_TRUE(tile.colIsValid(1));
  EXPECT_TRUE(tile.rowIsValid(2));
  EXPECT_TRUE(tile.colIsValid(2));
  EXPECT_FALSE(tile.rowIsValid(3));
  EXPECT_FALSE(tile.colIsValid(3));

  // copy over
  for (auto i = startRow; i < startRow + theight; i++) {
    for (auto j = startCol; j < startCol + twidth; j++) {
      EXPECT_TRUE(tile.rowIsValid(i));
      EXPECT_TRUE(tile.colIsValid(j));
      EXPECT_TRUE(img.rowIsValid(i));
      EXPECT_TRUE(img.colIsValid(j));
      tile.at(i, j) = img.at(i, j);
    }
  }

  // inspection
  EXPECT_EQ(tile.at(1, 1), 6);
  EXPECT_EQ(tile.at(1, 2), 7);
  EXPECT_EQ(tile.at(2, 1), 10);
  EXPECT_EQ(tile.at(2, 2), 11);

  EXPECT_EQ(tile.rowEnd(1), 7);
  EXPECT_EQ(tile.rowEnd(2), 11);
  EXPECT_EQ(tile.colEnd(1), 10);
  EXPECT_EQ(tile.colEnd(2), 11);
}

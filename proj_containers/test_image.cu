#include "containers/image.cuh"
#include "gtest/gtest.h"

TEST(ContainersImage, BasicChecks) {
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

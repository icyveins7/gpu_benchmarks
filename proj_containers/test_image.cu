#include "containers/image.cuh"
#include "thrust/sequence.h"
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

template <typename T, int TILE_DIM_X, int TILE_DIM_Y>
__global__ void testImageTileShmemFill(const containers::Image<const T> in,
                                       containers::Image<T> out,
                                       const int2 tileOffset) {
  if (blockIdx.x > 0 || blockIdx.y > 0)
    return;

  // just the first block
  __shared__ T s_data[TILE_DIM_X * TILE_DIM_Y];
  auto tile = containers::ImageTile<T>(
      s_data, (unsigned int)TILE_DIM_X, (unsigned int)TILE_DIM_Y,
      (unsigned int)tileOffset.y, (unsigned int)tileOffset.x);
  tile.fillFromImage(in);
  __syncthreads();

  // write back to global
  for (int ty = threadIdx.y; ty < TILE_DIM_Y; ty += blockDim.y) {
    int y = ty + tileOffset.y;
    for (int tx = threadIdx.x; tx < TILE_DIM_X; tx += blockDim.x) {
      int x = tx + tileOffset.x;

      T val = tile.at(y, x);
      out.at(ty, tx) = val;
    }
  }
}

TEST(ContainersImageTile, InKernelShmemFill) {
  containers::DeviceImageStorage<int> d_in(4, 4);
  thrust::sequence(d_in.vec.begin(), d_in.vec.end());
  thrust::host_vector<int> h_in = d_in.vec;
  for (int i = 0; i < (int)h_in.size(); ++i) {
    EXPECT_EQ(h_in[i], i);
  }

  containers::DeviceImageStorage<int> d_out(2, 2);

  int2 tileOffset{1, 1};
  dim3 tpb(32, 4);
  testImageTileShmemFill<int, 2, 2>
      <<<1, tpb>>>(d_in.cimage(), d_out.image(), tileOffset);

  thrust::host_vector<int> h_out = d_out.vec;
  EXPECT_EQ(h_out[0], 5);
  EXPECT_EQ(h_out[1], 6);
  EXPECT_EQ(h_out[2], 9);
  EXPECT_EQ(h_out[3], 10);
}

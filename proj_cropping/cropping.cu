#include "cropping.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

int main(int argc, char *argv[]) {
  std::printf("Cropping tests.\n");

  if (argc != 1 && argc != 9) {
    std::printf("Usage: %s <srcWidth> <srcHeight> <ptCoordX> <ptCoordY> "
                "<-x> <+x> <-y> <+y>\n",
                argv[0]);
    return 0;
  }

  // Simple defaults
  int srcWidth = 32, srcHeight = 32;
  int2 ptCoords;
  ptCoords.x = 15;
  ptCoords.y = 15;

  int4 cropDirections;
  cropDirections.x = 8;
  cropDirections.y = 7; // -8, +7 in x
  cropDirections.z = 8;
  cropDirections.w = 7; // -8, +7 in y

  if (argc == 9) {
    srcWidth = atoi(argv[1]);
    srcHeight = atoi(argv[2]);
    ptCoords.x = atoi(argv[3]);
    ptCoords.y = atoi(argv[4]);
    cropDirections.x = atoi(argv[5]);
    cropDirections.y = atoi(argv[6]);
    cropDirections.z = atoi(argv[7]);
    cropDirections.w = atoi(argv[8]);
  }
  // ============================================================

  // Make some simple source and dest
  int dstWidth = cropDirections.x + cropDirections.y + 1;
  int dstHeight = cropDirections.z + cropDirections.w + 1;

  thrust::host_vector<int> src(srcWidth * srcHeight);
  thrust::host_vector<int> dst(dstWidth * dstHeight);

  // Fill simply
  thrust::sequence(src.begin(), src.end());

  thrust::device_vector<int> d_src = src;
  thrust::device_vector<int> d_dst(dst.size());

  int thds = 32;
  int blks = dst.size() / thds + 1;
  cropAroundPoint_gridStrideKernel<<<blks, thds>>>(
      d_src.data().get(), srcWidth, srcHeight, cropDirections, ptCoords,
      d_dst.data().get());

  // Print results
  dst = d_dst;

  std::printf("Src\n");
  for (int i = 0; i < srcHeight; i++) {
    for (int j = 0; j < srcWidth; j++) {
      std::printf("%3d ", src[i * srcWidth + j]);
    }
    std::printf("\n");
  }

  std::printf("Dest around: %d -%d,+%d, %d -%d,+%d\n", ptCoords.x,
              cropDirections.x, cropDirections.y, ptCoords.y, cropDirections.z,
              cropDirections.w);
  for (int i = 0; i < dstHeight; i++) {
    for (int j = 0; j < dstWidth; j++) {
      std::printf("%3d ", dst[i * dstWidth + j]);
    }
    std::printf("\n");
  }

  return 0;
}

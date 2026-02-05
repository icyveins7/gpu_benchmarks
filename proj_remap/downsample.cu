#include "downsampling.cuh"
#include <iostream>
#include <numeric>
#include <vector>

void test(int width, int height, int ratio) {
  CentreAlignedDownsampler<> ds(width, height, ratio, ratio);
  printf("output %d x %d\n", ds.outputHeight(), ds.outputWidth());
  std::vector<int> x(width * height);
  std::iota(x.begin(), x.end(), 0);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      printf("%3d ", x[i * width + j]);
    }
    printf("\n");
  }
  printf("---\n");

  std::vector<int> y(ds.outputWidth() * ds.outputHeight());
  ds.downsampleImageOnHost(x.data(), y.data());

  for (int i = 0; i < ds.outputHeight(); ++i) {
    for (int j = 0; j < ds.outputWidth(); ++j) {
      printf("%3d ", y[i * ds.outputWidth() + j]);
    }
    printf("\n");
  }
}

int main() {
  test(4, 4, 2);
  // expect 0, 2
  test(9, 9, 3);
  // expect 1, 4, 7
  test(9, 9, 4);
  // expect 0, 4, 8
  return 0;
}

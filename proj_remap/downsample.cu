#include "downsampling.cuh"
#include "timer.h"
#include <iostream>
#include <numeric>
#include <vector>

void test(int width, int height, int ratio) {
  CentreAlignedDownsampler<> ds(width, height, ratio, ratio);
  printf("output %d x %d\n", ds.outputHeight(), ds.outputWidth());
  printf("input centre at %.1f, %.1f\n", ds.trueCentreX(), ds.trueCentreY());
  printf("output centre at %.1f, %.1f\n", ds.trueDownsampledCentreX(),
         ds.trueDownsampledCentreY());
  std::vector<int> x(width * height);
  std::iota(x.begin(), x.end(), 0);
  if (width < 100 && height < 100) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%3d ", x[i * width + j]);
      }
      printf("\n");
    }
    printf("---\n");
  }

  std::vector<int> y(ds.outputWidth() * ds.outputHeight());
  {
    HighResolutionTimer timer;
    ds.downsampleImageOnHost(x.data(), y.data());
  }

  if (width < 100 && height < 100) {
    for (int i = 0; i < ds.outputHeight(); ++i) {
      for (int j = 0; j < ds.outputWidth(); ++j) {
        printf("%3d ", y[i * ds.outputWidth() + j]);
      }
      printf("\n");
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc >= 3) {
    int length = atoi(argv[1]);
    int ratio = atoi(argv[2]);

    test(length, length, ratio);
  } else {
    test(4, 4, 2);
    // expect 0, 2
    test(9, 9, 3);
    // expect 1, 4, 7
    test(9, 9, 4);
    // expect 0, 4, 8
  }

  return 0;
}

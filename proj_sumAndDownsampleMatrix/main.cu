#include <iostream>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "sumAndDownsample.cuh"

void test(const size_t width, const size_t height, const size_t widthDsr,
          const size_t heightDsr) {

  if (width % widthDsr != 0 || height % heightDsr != 0) {
    throw std::invalid_argument("width and height must be divisible by "
                                "widthDsr and heightDsr respectively");
  }

  thrust::device_vector<float> img(width * height);
  thrust::sequence(img.begin(), img.end());

  thrust::device_vector<float> imgDsr(width / widthDsr * height / heightDsr);

  dim3 block(16, 16);
  dim3 grid((width / widthDsr + block.x - 1) / block.x,
            (height / heightDsr + block.y - 1) / block.y);
  printf("Grid: %dx%d Block: %dx%d\n", grid.x, grid.y, block.x, block.y);

  sumAndDownsampleMatrix<<<grid, block>>>(
      thrust::raw_pointer_cast(img.data()),
      thrust::raw_pointer_cast(imgDsr.data()), width, height, widthDsr,
      heightDsr);

  thrust::host_vector<float> h_imgDsr = imgDsr;
  thrust::host_vector<float> h_img = img;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%5.1f ", h_img[i * width + j]);
    }
    std::cout << std::endl;
  }
  std::cout << "|" << std::endl << "\\/" << std::endl;

  for (int i = 0; i < height / heightDsr; i++) {
    for (int j = 0; j < width / widthDsr; j++) {
      printf("%5.1f ", h_imgDsr[i * width / widthDsr + j]);
    }
    std::cout << std::endl;
  }

  printf("=======================\n");
}

int main(int argc, char *argv[]) {

  size_t width, height, widthDsr, heightDsr;

  {
    width = 4;
    height = 4;
    widthDsr = 2;
    heightDsr = 2;
    test(width, height, widthDsr, heightDsr);
  }
  {
    width = 6;
    height = 6;
    widthDsr = 3;
    heightDsr = 3;
    test(width, height, widthDsr, heightDsr);
  }
  {
    width = 64;
    height = 64;
    widthDsr = 2;
    heightDsr = 2;
    test(width, height, widthDsr, heightDsr);
  }

  return 0;
}

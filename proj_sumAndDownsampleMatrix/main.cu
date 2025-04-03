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

  // also make a vector to track passed thresholds
  const float threshold =
      static_cast<float>(img.size()) / 2.0f; // arbitrary value threshold
  thrust::device_vector<unsigned int> counter(1, 0);

  printf("Running simple kernel\n");
  sumAndDownsampleMatrix<<<grid, block>>>(
      thrust::raw_pointer_cast(img.data()),
      thrust::raw_pointer_cast(imgDsr.data()), width, height, widthDsr,
      heightDsr);

  printf("Running kernel with threshold with compiler-handled warp aggregated "
         "atomics\n");
  sumAndDownsampleMatrixWithThreshold<<<grid, block>>>(
      thrust::raw_pointer_cast(img.data()),
      thrust::raw_pointer_cast(imgDsr.data()), width, height, widthDsr,
      heightDsr, threshold, thrust::raw_pointer_cast(counter.data()));

  // reset counter
  counter[0] = 0;

  printf("Running kernel with threshold with manual warp aggregated "
         "atomics\n");
  sumAndDownsampleMatrixWithThreshold<float, true><<<grid, block>>>(
      thrust::raw_pointer_cast(img.data()),
      thrust::raw_pointer_cast(imgDsr.data()), width, height, widthDsr,
      heightDsr, threshold, thrust::raw_pointer_cast(counter.data()));

  thrust::host_vector<float> h_imgDsr = imgDsr;
  thrust::host_vector<float> h_img = img;
  thrust::host_vector<unsigned int> h_counter = counter;

  std::cout << "Threshold: " << threshold << std::endl;
  printf("Counter: %u / %zd\n", h_counter[0], imgDsr.size());
  if (h_imgDsr.size() < 100000) {
    // don't print the big matrices
    for (size_t i = 0; i < height; i++) {
      for (size_t j = 0; j < width; j++) {
        printf("%5.1f ", h_img[i * width + j]);
      }
      std::cout << std::endl;
    }
    std::cout << "|" << std::endl << "\\/" << std::endl;

    for (size_t i = 0; i < height / heightDsr; i++) {
      for (size_t j = 0; j < width / widthDsr; j++) {
        printf("%5.1f ", h_imgDsr[i * width / widthDsr + j]);
      }
      std::cout << std::endl;
    }
  }

  printf("=======================\n");
}

int main() {

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
  {
    // make really large one
    width = 8192;
    height = 8192;
    widthDsr = 4;
    heightDsr = 4;
    test(width, height, widthDsr, heightDsr);
  }

  return 0;
}

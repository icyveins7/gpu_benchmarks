#include "sumAndDownsample.cuh"
#include <cstdint>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

template <typename Tdata, typename Tidx, typename Tsqueeze>
void test(const size_t width, const size_t height, const size_t widthDsr,
          const size_t heightDsr, const size_t batch, bool print = false) {

  thrust::device_vector<Tdata> img(batch * width * height);
  thrust::sequence(img.begin(), img.end());
  size_t oWidth = width / widthDsr;
  size_t oHeight = height / heightDsr;
  printf("oWidth: %zu, oHeight: %zu\n", oWidth, oHeight);
  thrust::device_vector<Tdata> dsrImg(batch * oWidth * oHeight);
  thrust::device_vector<Tsqueeze> maxArgmax(batch);

  dim3 tpb(32, 4);
  dim3 blks((width / widthDsr + tpb.x - 1) / tpb.x,
            (height / heightDsr + tpb.y - 1) / tpb.y, batch);
  printf("Grid: %dx%dx%d Block: %dx%d\n", blks.x, blks.y, blks.z, tpb.x, tpb.y);

  size_t shmemBytes =
      tpb.x * tpb.y * sizeof(Tdata) * (widthDsr * heightDsr + 1);
  sumAndDownsampleMatrixWithArgmaxBlockwiseKernel<Tdata, Tidx, Tsqueeze>
      <<<blks, tpb, shmemBytes>>>(img.data().get(), dsrImg.data().get(), width,
                                  height, batch, widthDsr, heightDsr,
                                  maxArgmax.data().get());

  thrust::host_vector<Tdata> h_dsrImg = dsrImg;
  thrust::host_vector<Tsqueeze> h_maxArgmax = maxArgmax;
  for (size_t i = 0; i < batch; i++) {
    Tdata max;
    Tidx argmax;
    unsqueezeValueIndex<Tdata, Tidx, Tsqueeze>(h_maxArgmax[i], max, argmax);
    printf("Batch %zu\n", i);
    std::cout << "max: " << max << " argmax: " << argmax << std::endl;
    std::cout << "argmax conversion: " << argmax / oWidth << " "
              << argmax % oWidth << std::endl;
  }

  if (print) {
    for (size_t i = 0; i < batch; i++) {
      printf("---------\n");
      for (size_t j = 0; j < oHeight; j++) {
        for (size_t k = 0; k < oWidth; k++) {
          printf("%d ", h_dsrImg[i * oWidth * oHeight + j * oWidth + k]);
        }
        printf("\n");
      }
    }
  }

  printf("=======================\n");
}

int main() {

  size_t width, height, widthDsr, heightDsr, batch;

  {
    width = 7;
    height = 7;
    widthDsr = 2;
    heightDsr = 2;
    batch = 2;
    test<uint16_t, uint16_t, unsigned int>(width, height, widthDsr, heightDsr,
                                           batch, true);
  }
  {
    width = 67;
    height = 67;
    widthDsr = 2;
    heightDsr = 2;
    batch = 2;
    test<uint16_t, uint16_t, unsigned int>(width, height, widthDsr, heightDsr,
                                           batch, false);
  }
  {
    width = 8192;
    height = 8192;
    widthDsr = 5;
    heightDsr = 5;
    batch = 8;
    test<float, unsigned int, unsigned long long int>(width, height, widthDsr,
                                                      heightDsr, batch, false);
  }
  return 0;
}

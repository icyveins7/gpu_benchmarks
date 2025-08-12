#include "sumAndDownsample.cuh"
#include <cstdint>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

template <typename Tdata, typename Tidx, typename Tsqueeze>
void test(const size_t width, const size_t height, const size_t widthDsr,
          const size_t heightDsr, const size_t batch) {

  thrust::device_vector<Tdata> img(batch * width * height);
  thrust::sequence(img.begin(), img.end());
  size_t oWidth = width / widthDsr;
  size_t oHeight = height / heightDsr;
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
  }

  printf("=======================\n");
}

int main() {

  size_t width, height, widthDsr, heightDsr, batch;

  {
    width = 4;
    height = 4;
    widthDsr = 2;
    heightDsr = 2;
    batch = 1;
    test<uint16_t, uint16_t, unsigned int>(width, height, widthDsr, heightDsr,
                                           batch);
  }
  return 0;
}

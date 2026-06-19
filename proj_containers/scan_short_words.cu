/*
 * NOTE: this entire file's premise is wrong!
 * Packed scan cannot be done this way. It will produce incorrect results.
 *
 */

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
#include <cstdint>
#include <cstdlib>
#include <iostream>

struct NaiveScanOp {
  __device__ int8_t operator()(int8_t a, int8_t b) { return b < a ? b : a; }
};

struct PackedScanOp {
  __device__ uint32_t operator()(uint32_t a, uint32_t b) {
    // this doesn't work! it's wrong
    // extract all bytes from b
    int8_t b0 = (b & 0x000000FF);
    int8_t b1 = (b & 0x0000FF00) >> 8;
    int8_t b2 = (b & 0x00FF0000) >> 16;
    int8_t b3 = (b & 0xFF000000) >> 24;

    // extract last byte from a
    int8_t a0 = (int8_t)(a & 0xFF);

    int8_t r0 = b0 < a0 ? b0 : a0;
    int8_t r1 = b1 < r0 ? b1 : r0;
    int8_t r2 = b2 < r1 ? b2 : r1;
    int8_t r3 = b3 < r2 ? b3 : r2;
    return (uint32_t)(r3 << 24) | (uint32_t)(r2 << 16) | (uint32_t)(r1 << 8) |
           (uint32_t)r0;
  }

  // __device__ uint32_t operator()(uint32_t a, uint32_t b) {
  //   // extract last byte from a
  //   int8_t a_end = a & 0xFF;
  //
  //   // custom blending
  //   uint32_t left = a_end << 24 | b >> 8;
  //   unsigned int mask = __vcmpgts4(b, left);
  //   return (mask & left) | (~mask & b);
  // }
};

int main(int argc, char *argv[]) {
  printf("Maximising scan speed\n");

  int width = 16384, height = 16384;
  if (argc >= 3) {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
  }
  printf("width: %d, height: %d\n", width, height);

  containers::PinnedHostImageStorage<int8_t> h_in(width, height);
  for (int i = 0; i < width * height; ++i) {
    h_in.vec[i] = std::rand() % 8;
  }
  if (width <= 8 && height <= 8) {
    auto in = h_in.cimage();
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%hhd ", in.at(i, j));
      }
      printf("\n");
    }
    printf("----------\n");
  }

  containers::DeviceImageStorage<int8_t> d_in(width, height);
  h_in.toDevice(d_in);
  containers::DeviceImageStorage<int8_t> d_out(width, height);

  using Tidx = int16_t;

  {
    auto rowKeys = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        cubw::helpers::IndexToRowFunctor<Tidx>{(Tidx)width});
    cubw::DeviceScan::InclusiveScanByKey<decltype(rowKeys), int8_t *, int8_t *,
                                         NaiveScanOp>
        scan(width * height);
    scan.exec(rowKeys, d_in.vec.data().get(), d_out.vec.data().get(),
              NaiveScanOp{}, width * height);

    if (width <= 8 && height <= 8) {
      d_out.toHost(h_in);
      cudaDeviceSynchronize();
      auto out = h_in.cimage();
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          printf("%hhd ", out.at(i, j));
        }
        printf("\n");
      }
      printf("----------\n");
    }
  }

  {
    using Tpacked = uint32_t;
    Tidx packedWidth = width / 4;
    auto rowKeys = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        cubw::helpers::IndexToRowFunctor<Tidx>{(Tidx)(packedWidth)});
    cubw::DeviceScan::InclusiveScanByKey<decltype(rowKeys), Tpacked *, int8_t *,
                                         PackedScanOp>
        scan(packedWidth * height);
    scan.exec(rowKeys, (Tpacked *)d_in.vec.data().get(), d_out.vec.data().get(),
              PackedScanOp{}, packedWidth * height);

    if (width <= 8 && height <= 8) {
      d_out.toHost(h_in);
      cudaDeviceSynchronize();
      auto out = h_in.cimage();
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          printf("%hhd ", out.at(i, j));
        }
        printf("\n");
      }
      printf("----------\n");
    }
  }

  return 0;
}

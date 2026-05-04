#undef NDEBUG
#include <iostream>

#include "containers/image.cuh"

#include <thrust/device_vector.h>

__global__ void testkernel(const int *a) { printf("%d\n", a[0]); }
__global__ void testImagekernel(containers::Image<const int> a) {
  printf("%u * %u\n", a.width, a.height);
}

int main(int argc, char *argv[]) {
  printf("Testing stream ordered allocators (device vector comparison)\n");
  int len = 1;
  if (argc > 1) {
    len = atoi(argv[1]);
  }
  printf("Using length %d\n", len);

  using Vec = thrust::device_vector<int>;
  using ImgVec = containers::DeviceImageStorage<int>;

  for (int i = 0; i < 3; ++i) {
    Vec d_a(len * (i + 1));
    testkernel<<<1, 1, 0>>>(d_a.data().get());
    Vec d_b(len * (i + 1));
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data().get(), d_a.data().get(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice);
#endif
  }
  for (int i = 0; i < 3; ++i) {
    Vec d_a(len);
    testkernel<<<1, 1, 0>>>(d_a.data().get());
    Vec d_b(len);
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data().get(), d_a.data().get(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice);
#endif
  }

  for (int i = 0; i < 3; ++i) {
    ImgVec d_a(len * (i + 1), len * (i + 1));
    testImagekernel<<<1, 1, 0>>>(d_a.cimage());
    ImgVec d_b(len * (i + 1), len * (i + 1));
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data().get(), d_a.data().get(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice);
#endif
  }
  for (int i = 0; i < 3; ++i) {
    ImgVec d_a(len, len);
    testImagekernel<<<1, 1, 0>>>(d_a.cimage());
    ImgVec d_b(len, len);
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data().get(), d_a.data().get(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice);
#endif
  }

  return 0;
}

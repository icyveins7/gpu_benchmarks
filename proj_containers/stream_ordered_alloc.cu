#undef NDEBUG
#include <iostream>

#include "containers/image.cuh"
#include "containers/streams.cuh"

__global__ void testkernel(const int *a) { printf("%d\n", a[0]); }
__global__ void testImagekernel(containers::Image<const int> a) {
  printf("%u * %u\n", a.width, a.height);
}

int main(int argc, char *argv[]) {
  printf("Testing stream ordered allocators\n");
  int len = 1;
  if (argc > 1) {
    len = atoi(argv[1]);
  }
  printf("Using length %d\n", len);

  containers::CudaStream stream;

  for (int i = 0; i < 3; ++i) {
    containers::StreamOrderedDeviceStorage<int> d_a(len * (i + 1), stream());
    testkernel<<<1, 1, 0, stream()>>>(d_a.data());
    containers::StreamOrderedDeviceStorage<int> d_b(len * (i + 1), stream());
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data(), d_a.data(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice, stream());
#endif
  }
  for (int i = 0; i < 3; ++i) {
    containers::StreamOrderedDeviceStorage<int> d_a(len, stream());
    testkernel<<<1, 1, 0, stream()>>>(d_a.data());
    containers::StreamOrderedDeviceStorage<int> d_b(len, stream());
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data(), d_a.data(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice, stream());
#endif
  }

  for (int i = 0; i < 3; ++i) {
    containers::StreamOrderedDeviceImageStorage<int> d_a(
        len * (i + 1), len * (i + 1), stream());
    testImagekernel<<<1, 1, 0, stream()>>>(d_a.cimage());
    containers::StreamOrderedDeviceImageStorage<int> d_b(
        len * (i + 1), len * (i + 1), stream());
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data(), d_a.data(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice, stream());
#endif
  }
  for (int i = 0; i < 3; ++i) {
    containers::StreamOrderedDeviceImageStorage<int> d_a(len, len, stream());
    testImagekernel<<<1, 1, 0, stream()>>>(d_a.cimage());
    containers::StreamOrderedDeviceImageStorage<int> d_b(len, len, stream());
#if defined(WITH_MEMCPY)
    cudaMemcpyAsync(d_b.data(), d_a.data(), sizeof(int) * len,
                    cudaMemcpyDeviceToDevice, stream());
#endif
  }

  stream.sync();

  return 0;
}

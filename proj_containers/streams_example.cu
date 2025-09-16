#include "containers.cuh"
#include "pinnedalloc.cuh"
#include <thrust/device_vector.h>

template <typename T> __global__ void simpleKernel(T *x, int len) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < len;
       i += gridDim.x * blockDim.x)
    for (int j = 0; j < 1024; ++j)
      x[i] = x[i] * x[i];
}

int main() {
  size_t len = 1000000;
  thrust::pinned_host_vector<float> h_data(len);
  thrust::device_vector<float> d_data(len);
  thrust::device_vector<float> d_data2(len);

  // Run on separate streams
  {
    containers::CudaStream compute_stream(cudaStreamNonBlocking);
    containers::CudaStream copy_stream(cudaStreamNonBlocking);
    // Warmup?
    // NOTE: on some systems, without this call, the first kernel launch is not
    // async, and will only occur after the memcpyasync completes
    // See the following link:
    // https://stackoverflow.com/questions/72012121/why-the-first-cuda-kernel-cannot-overlap-with-previous-memcpy
    simpleKernel<<<256, 256, 0, compute_stream()>>>(d_data2.data().get(), 1);

    compute_stream.sync();
    copy_stream.sync();

    for (int i = 0; i < 3; ++i) {
      cudaMemcpyAsync(d_data.data().get(), h_data.data().get(),
                      h_data.size() * sizeof(float), cudaMemcpyHostToDevice,
                      copy_stream());
      simpleKernel<<<256, 256, 0, compute_stream()>>>(d_data2.data().get(),
                                                      d_data2.size());
      compute_stream.sync();
      copy_stream.sync();
    }
  }

  return 0;
}

#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cuComplex.h>
#include <nvtx3/nvToolsExt.h>

#ifndef BASICOPS_TYPE
#define BASICOPS_TYPE int
#endif

// cuComplex arithmetic operators so thrust::plus/multiplies work with it
__host__ __device__ inline cuComplex operator+(cuComplex a, cuComplex b) {
  return cuCaddf(a, b);
}
__host__ __device__ inline cuComplex operator*(cuComplex a, cuComplex b) {
  return cuCmulf(a, b);
}

// -1 fill value, specialised for types that can't be constructed from int
template <typename T> T minus_one() { return T(-1); }
template <> inline cuComplex minus_one<cuComplex>() {
  return make_cuComplex(-1.0f, 0.0f);
}

int main(int argc, char* argv[]) {
  using T = BASICOPS_TYPE;

  size_t cols = 0, rows = 0;

  if (argc == 2) {
    cols = std::atoi(argv[1]);
    printf("cols = %zu\n", cols);
    if (cols == 0) {
      std::cerr << "Usage: " << argv[0] << " <cols>" << std::endl;
      return 1;
    }

  } else if (argc == 3) {
    rows = std::atoi(argv[1]);
    cols = std::atoi(argv[2]);
    printf("rows = %zu, cols = %zu\n", rows, cols);
    if (rows == 0 || cols == 0) {
      std::cerr << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
      return 1;
    }
  } else {
    std::cerr << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
    std::cerr << "Usage: " << argv[0] << " <cols>" << std::endl;
    return 1;
  }

  // Setup
  thrust::host_vector<T> h_a(rows * cols);
  thrust::device_vector<T> d_a(rows * cols);
  thrust::host_vector<T> h_b(rows * cols);
  thrust::device_vector<T> d_b(rows * cols);
  thrust::host_vector<T> h_c(rows * cols);
  thrust::device_vector<T> d_c(rows * cols);

  for (int i = 0; i < 3; ++i) {
    // 1a. Fill with 0s with cudaMemset
    nvtxRangePushA("cudaMemset");
    cudaMemset(d_a.data().get(), 0, rows * cols * sizeof(T));
    nvtxRangePop();
    // 1b. Fill with 0s using thrust fill
    nvtxRangePushA("thrust_fill_0");
    thrust::fill(d_a.begin(), d_a.end(), T{});
    nvtxRangePop();
    // 2. Fill with -1 values using thrust fill
    // (cannot do this via memset since not byte-valued)
    nvtxRangePushA("thrust_fill_minus1");
    thrust::fill(d_b.begin(), d_b.end(), minus_one<T>());
    nvtxRangePop();

    // 3a. Copy from a to b via thrust
    nvtxRangePushA("thrust_copyDtoD");
    thrust::copy(d_a.begin(), d_a.end(), d_b.begin());
    nvtxRangePop();
    // 3b. Copy from a to b via cudaMemcpy
    nvtxRangePushA("cudaMemcpyDtoD");
    cudaMemcpy(d_b.data().get(), d_a.data().get(), rows * cols * sizeof(T),
               cudaMemcpyDeviceToDevice);
    nvtxRangePop();

    // 4a. Add A and B in-place
    nvtxRangePushA("thrust_transform_add_inplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_a.begin(),
                      thrust::plus<T>());
    nvtxRangePop();
    // 4b. Add A and B out-of-place
    nvtxRangePushA("thrust_transform_add_outofplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                      thrust::plus<T>());
    nvtxRangePop();

    // 5a. Multiply A and B in-place
    nvtxRangePushA("thrust_transform_mul_inplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_a.begin(),
                      thrust::multiplies<T>());
    nvtxRangePop();
    // 5b. Multiply A and B out-of-place
    nvtxRangePushA("thrust_transform_mul_outofplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                      thrust::multiplies<T>());
    nvtxRangePop();

    // End. copy out to prevent possible optimizations away
    thrust::copy(d_a.begin(), d_a.end(), h_a.begin());
    thrust::copy(d_b.begin(), d_b.end(), h_b.begin());
  }

  return 0;
}

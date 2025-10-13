#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <nvtx3/nvToolsExt.h>

int main(int argc, char *argv[]) {
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
  thrust::host_vector<int> h_a(rows * cols);
  thrust::device_vector<int> d_a(rows * cols);
  thrust::host_vector<int> h_b(rows * cols);
  thrust::device_vector<int> d_b(rows * cols);
  thrust::host_vector<int> h_c(rows * cols);
  thrust::device_vector<int> d_c(rows * cols);

  for (int i = 0; i < 3; ++i) {
    // 1a. Fill with 0s with cudaMemset
    nvtxRangePushA("cudaMemset");
    cudaMemset(d_a.data().get(), 0, rows * cols * sizeof(int));
    nvtxRangePop();
    // 1b. Fill with 0s using thrust fill
    nvtxRangePushA("thrust_fill_0");
    thrust::fill(d_a.begin(), d_a.end(), 0);
    nvtxRangePop();
    // 2. Fill with -1 values using thrust fill
    // (cannot do this via memset since not byte-valued)
    nvtxRangePushA("thrust_fill_minus1");
    thrust::fill(d_b.begin(), d_b.end(), -1);
    nvtxRangePop();

    // 3a. Copy from a to b via thrust
    nvtxRangePushA("thrust_copyDtoD");
    thrust::copy(d_a.begin(), d_a.end(), d_b.begin());
    nvtxRangePop();
    // 3b. Copy from a to b via cudaMemcpy
    nvtxRangePushA("cudaMemcpyDtoD");
    cudaMemcpy(d_b.data().get(), d_a.data().get(), rows * cols * sizeof(int),
               cudaMemcpyDeviceToDevice);
    nvtxRangePop();

    // 4a. Add A and B in-place
    nvtxRangePushA("thrust_transform_add_inplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_a.begin(),
                      thrust::plus<int>());
    nvtxRangePop();
    // 4b. Add A and B out-of-place
    nvtxRangePushA("thrust_transform_add_outofplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                      thrust::plus<int>());
    nvtxRangePop();

    // 5a. Multiply A and B in-place
    nvtxRangePushA("thrust_transform_mul_inplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_a.begin(),
                      thrust::multiplies<int>());
    nvtxRangePop();
    // 5b. Multiply A and B out-of-place
    nvtxRangePushA("thrust_transform_mul_outofplace");
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                      thrust::multiplies<int>());
    nvtxRangePop();

    // End. copy out to prevent possible optimizations away
    thrust::copy(d_a.begin(), d_a.end(), h_a.begin());
    thrust::copy(d_b.begin(), d_b.end(), h_b.begin());
  }

  return 0;
}

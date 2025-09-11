#include "wccl_kernels.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  constexpr int rows = 4;
  constexpr int cols = 5;
  // clang-format off
  const std::vector<uint8_t> input = {
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 0, 1,
  };
  if (input.size() != rows * cols) {
    throw std::runtime_error("input size is not equal to rows * cols");
  }
  // clang-format on
  thrust::host_vector<uint8_t> h_input(rows * cols);
  thrust::copy(input.begin(), input.end(), h_input.begin());
  thrust::device_vector<uint8_t> d_input(rows * cols);
  thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
  // Copy back to check?
  thrust::copy(d_input.begin(), d_input.end(), h_input.begin());
  for (int i = 0; i < h_input.size(); ++i)
    printf("%d ", h_input[i]);
  printf("\n");

  thrust::device_vector<int16_t> d_mapping_vec(rows * cols);

  // Encapsulate
  wccl::DeviceImage<uint8_t> d_image(d_input, rows, cols);
  wccl::DeviceImage<short> d_mapping(d_mapping_vec, rows, cols);

  // Kernel
  dim3 tpb(32, 4);
  dim3 bpg(1, 1);
  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  size_t shmem = tileDims.x * tileDims.y * 2 * sizeof(short);
  printf("Launching kernel with shmem = %zu\n", shmem);
  wccl::connect_kernel<short>
      <<<bpg, tpb, shmem>>>(d_image, d_mapping, tileDims, windowDist);

  thrust::host_vector<short> h_mapping = d_mapping_vec;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      if (h_mapping[r * cols + c] < 0)
        printf("-- ");
      else
        printf("%2d ", h_mapping[r * cols + c]);
    }
    printf("\n");
  }

  return 0;
}

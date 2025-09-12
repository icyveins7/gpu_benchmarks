#include "wccl.h"
#include "wccl_kernels.cuh"

#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  // ================= Example 1
  // constexpr int rows = 4;
  // constexpr int cols = 5;
  // // clang-format off
  // const std::vector<uint8_t> input = {
  //   1, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   0, 0, 1, 0, 0,
  //   0, 1, 0, 0, 1,
  // };
  // // clang-format on
  // // ================= Example 2
  // constexpr int rows = 8;
  // constexpr int cols = 5;
  // // clang-format off
  // const std::vector<uint8_t> input = {
  //   1, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   0, 0, 1, 0, 0,
  //   0, 1, 0, 0, 1,
  //   0, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   1, 0, 0, 0, 0,
  //   0, 0, 0, 0, 0,
  // };
  // // clang-format on
  // // ================= Example 3
  // constexpr int rows = 8192;
  // constexpr int cols = 1024;
  // // clang-format off
  // std::vector<uint8_t> input(rows * cols);
  // for (size_t i = 0; i < rows * cols; ++i)
  //   input[i] = rand() % 2;
  // // clang-format on
  // ================= Example 4
  constexpr int rows = 8;
  constexpr int cols = 64;
  // clang-format off
  std::vector<uint8_t> input(rows * cols);
  for (size_t i = 0; i < rows * cols; ++i)
    input[i] = rand() % 2;
  // clang-format on

  typedef short2 KernelColRowType;
  typedef int MappingType;

  if (input.size() != rows * cols) {
    throw std::runtime_error("input size is not equal to rows * cols");
  }
  thrust::host_vector<uint8_t> h_input(rows * cols);
  thrust::copy(input.begin(), input.end(), h_input.begin());
  thrust::device_vector<uint8_t> d_input(rows * cols);
  thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
  // Copy back to check?
  thrust::copy(d_input.begin(), d_input.end(), h_input.begin());

  thrust::device_vector<MappingType> d_mapping_vec(rows * cols);

  // Encapsulate
  wccl::DeviceImage<uint8_t> d_image(d_input, rows, cols);
  wccl::DeviceImage<MappingType> d_mapping(d_mapping_vec, rows, cols);

  // Kernel
  dim3 tpb(32, 4);
  dim3 bpg(d_image.width / tpb.x + (d_image.width % tpb.x ? 1 : 0),
           d_image.height / tpb.y + (d_image.height % tpb.y ? 1 : 0));
  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  size_t shmem = tileDims.x * tileDims.y * 2 * sizeof(short);
  printf("Launching (%d, %d) blks kernel with shmem = %zu\n", bpg.x, bpg.y,
         shmem);
  wccl::connect_kernel<MappingType, KernelColRowType>
      <<<bpg, tpb, shmem>>>(d_image, d_mapping, tileDims, windowDist);

  thrust::host_vector<MappingType> h_mapping_vec = d_mapping_vec;

  if (rows <= 64 && cols <= 64)
    printf("%s\n",
           wccl::prettystring<MappingType>(
               thrust::raw_pointer_cast(h_mapping_vec.data()), rows, cols)
               .c_str());

  return 0;
}

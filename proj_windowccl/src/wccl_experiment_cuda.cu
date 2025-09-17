#include "wccl.h"
#include "wccl_kernels.cuh"

#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "pinnedalloc.cuh"

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

  // ================= Example 3 (for timing)
  constexpr int rows = 8192;
  constexpr int cols = 1024;
  // clang-format off
  std::vector<uint8_t> input(rows * cols);
  for (size_t i = 0; i < rows * cols; ++i)
    input[i] = rand() % 2;
  const int2 tileDims = {64, 8};
  // const int2 tileDims = {64, 16};
  // const int2 tileDims = {32, 8};
  // const int2 tileDims = {32, 4};
  // clang-format on

  // // ================= Example 4
  // constexpr int rows = 8;
  // constexpr int cols = 64;
  // // clang-format off
  // std::vector<uint8_t> input(rows * cols);
  // for (size_t i = 0; i < rows * cols; ++i)
  //   input[i] = rand() % 2;
  // const int2 tileDims = {32, 4};
  // // clang-format on

  // // ================= Example 5
  // constexpr int rows = 12;
  // constexpr int cols = 6;
  // // clang-format off
  // const std::vector<uint8_t> input = {
  //   0, 0, 1, 0, 1, 0,
  //   0, 1, 0, 0, 0, 1,
  //   0, 1, 0, 0, 0, 1,
  //   0, 0, 1, 0, 1, 0,
  //   0, 0, 1, 0, 1, 0,
  //   0, 1, 0, 0, 0, 1,
  //   0, 1, 0, 0, 0, 1,
  //   0, 0, 1, 0, 1, 0,
  //   1, 0, 0, 1, 0, 0,
  //   1, 0, 1, 0, 1, 0,
  //   0, 1, 0, 0, 0, 0,
  //   0, 0, 0, 0, 0, 0,
  // };
  // const int2 tileDims = {32, 4};
  // // clang-format on

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

  // ========== Kernel 1. Local tile merge (using non atomics)
#ifdef USE_ATOMICFREE_LOCAL
  printf("Using atomic-free local merge\n");
  dim3 tpb(32, 4);
  if (tpb.x > tileDims.x || tpb.y > tileDims.y) {
    throw std::runtime_error("tpb exceeds tileDims");
  }
  dim3 bpg(d_image.width / tileDims.x + (d_image.width % tileDims.x ? 1 : 0),
           d_image.height / tileDims.y + (d_image.height % tileDims.y ? 1 : 0));
  const int2 windowDist = {1, 1};
  // NOTE: for now on Linux or at least <CUDA 12.9, putting wrong shared mem
  // size does not seem to trigger compute-sanitizer errors, so be warned!
  size_t shmem = tileDims.x * tileDims.y * 2 * sizeof(KernelColRowType);
  printf("Launching (%d, %d) blks (%d, %d) threads kernel with shmem = % zu\n ",
         bpg.x, bpg.y, tpb.x, tpb.y, shmem);
  wccl::local_connect_kernel<MappingType, KernelColRowType>
      <<<bpg, tpb, shmem>>>(d_image, d_mapping, tileDims, windowDist);
#else
  // ========= Kernel 1. Local tile merge (using atomics)
  dim3 tpb(32, 4);
  if (tpb.x > tileDims.x || tpb.y > tileDims.y) {
    throw std::runtime_error("tpb exceeds tileDims");
  }
  dim3 bpg(d_image.width / tileDims.x + (d_image.width % tileDims.x ? 1 : 0),
           d_image.height / tileDims.y + (d_image.height % tileDims.y ? 1 : 0));
  const int2 windowDist = {1, 1};

  size_t shmem =
      tileDims.x * tileDims.y *
          (sizeof(MappingType) * 2) + // tile + active index bookkeeping
      2 * sizeof(
              unsigned int); // counter for atomics + counter for active indices
  printf("Launching (%d, %d) blks (%d, %d) threads kernel (atomics) with shmem "
         "= %zu\n",
         bpg.x, bpg.y, tpb.x, tpb.y, shmem);
  wccl::local_connect_naive_unionfind_kernel<MappingType>
      <<<bpg, tpb, shmem>>>(d_image, d_mapping, tileDims, windowDist);
#endif

  // Pull data and check
  thrust::host_vector<MappingType> h_mapping_vec = d_mapping_vec;

  if (rows <= 64 && cols <= 64) {
    printf("%s\n", wccl::idxstring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, "%2d ", "%2c ", tileDims.x, tileDims.y)
                       .c_str());

    printf("%s\n", wccl::prettystring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, tileDims.x, tileDims.y)
                       .c_str());
  }

  // Kernel 2. Cross-tile merge
  thrust::pinned_host_vector<unsigned int> h_counter(1);
  h_counter[0] = 0;
  thrust::device_vector<unsigned int> d_counter(1);
  thrust::copy(h_counter.begin(), h_counter.end(), d_counter.begin());
  unsigned int prevCounter;
  size_t numUnionFindIters = 0;

  do {
    prevCounter = h_counter[0];
    wccl::naive_global_unionfind_kernel<MappingType>
        <<<bpg, tpb>>>(d_mapping, tileDims, windowDist, d_counter.data().get());
    thrust::copy(d_counter.begin(), d_counter.end(), h_counter.begin());
    // printf("Update count: %u\n", h_counter[0]);
    numUnionFindIters++;

    // We can print to see it evolve
    if (rows <= 64 && cols <= 64) {
      h_mapping_vec = d_mapping_vec;
      printf("%s\n ========================== \n",
             wccl::idxstring<MappingType>(
                 thrust::raw_pointer_cast(h_mapping_vec.data()), rows, cols,
                 "%2d ", "%2c ", tileDims.x, tileDims.y)
                 .c_str());
    }
  } while (prevCounter != h_counter[0]);

  printf("numUnionFindIters = %zu\n", numUnionFindIters);
  h_mapping_vec = d_mapping_vec;

  if (rows <= 64 && cols <= 64) {
    printf("%s\n ========================== \n",
           wccl::idxstring<MappingType>(
               thrust::raw_pointer_cast(h_mapping_vec.data()), rows, cols,
               "%2d ", "%2c ", tileDims.x, tileDims.y)
               .c_str());
    printf("%s\n", wccl::prettystring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, tileDims.x, tileDims.y)
                       .c_str());
  }
  return 0;
}

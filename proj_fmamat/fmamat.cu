#include "../include/pinnedalloc.cuh"
#include "fmamat.h"
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

int main(int argc, char *argv[]) {
  printf("FMA across a matrix\n");
  printf("Usage: %s [width] [height] [repeats]\n", argv[0]);

  int width = 8, height = 8;
  int repeats = 2;
  // Parse from command line
  if (argc > 1) {
    width = atoi(argv[1]);
  }
  if (argc > 2) {
    height = atoi(argv[2]);
  }
  if (argc > 3) {
    repeats = atoi(argv[3]);
  }
  printf("width: %d, height: %d\n", width, height);
  printf("repeats: %d\n", repeats);

  using Tx = uint16_t;
  using Ty = uint16_t;

  thrust::pinned_host_vector<Tx> src(width * height);
  thrust::pinned_host_vector<Ty> dst(width * height);
  thrust::pinned_host_vector<float> m(width);
  thrust::pinned_host_vector<float> c(width);

  // Some initial values
  thrust::sequence(src.begin(), src.end(), 1.0f);
  thrust::sequence(m.begin(), m.end(), 1.0f);
  thrust::sequence(c.begin(), c.end(), 0.1f, 0.1f);

  thrust::device_vector<Tx> d_src = src;
  thrust::device_vector<Ty> d_dst = dst;
  thrust::device_vector<float> d_m = m;
  thrust::device_vector<float> d_c = c;

  // Change interrim calculation type here
  using Tcalc = float;

  // ======= Naive kernel implementation
  {
    dim3 threads_per_blk(128);
    dim3 blks_per_grid(d_src.size() / threads_per_blk.x + 1);

    nvtxRangePush("naive kernel");
    for (int i = 0; i < repeats; i++)
      naive_fmamat_columns_kernel<Tcalc><<<blks_per_grid, threads_per_blk>>>(
          d_src.data().get(), height, width, d_m.data().get(), d_c.data().get(),
          d_dst.data().get());

    cudaDeviceSynchronize();
    nvtxRangePop();
  }
  // ======= Shared mem kernel implementations
  {
    int rows_per_blk = 2;
    dim3 threads_per_blk(32, rows_per_blk);
    dim3 blks_per_grid(width / threads_per_blk.x + 1,
                       height / rows_per_blk + 1);
    const int smem_size = 64 * sizeof(Tcalc);

    nvtxRangePush("shared mem kernel, 2x32");
    for (int i = 0; i < repeats; i++)
      shared_fmamat_columns_kernel<Tcalc>
          <<<blks_per_grid, threads_per_blk, smem_size>>>(
              d_src.data().get(), height, width, d_m.data().get(),
              d_c.data().get(), d_dst.data().get());
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
  {
    int rows_per_blk = 8;
    dim3 threads_per_blk(32, rows_per_blk);
    dim3 blks_per_grid(width / threads_per_blk.x + 1,
                       height / rows_per_blk + 1);
    const int smem_size = 64 * sizeof(Tcalc);

    nvtxRangePush("shared mem kernel, 8x32");
    for (int i = 0; i < repeats; i++)
      shared_fmamat_columns_kernel<Tcalc>
          <<<blks_per_grid, threads_per_blk, smem_size>>>(
              d_src.data().get(), height, width, d_m.data().get(),
              d_c.data().get(), d_dst.data().get());
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
  {
    int rows_per_blk = 16;
    dim3 threads_per_blk(32, rows_per_blk);
    dim3 blks_per_grid(width / threads_per_blk.x + 1,
                       height / rows_per_blk + 1);
    const int smem_size = 64 * sizeof(Tcalc);

    nvtxRangePush("shared mem kernel, 16x32");
    for (int i = 0; i < repeats; i++)
      shared_fmamat_columns_kernel<Tcalc>
          <<<blks_per_grid, threads_per_blk, smem_size>>>(
              d_src.data().get(), height, width, d_m.data().get(),
              d_c.data().get(), d_dst.data().get());

    cudaDeviceSynchronize();
    nvtxRangePop();
  }

  if (width * height <= 64) {
    // Print output
    dst = d_dst;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        printf("%8.3f ", (double)dst[i * width + j]);
      }
      std::cout << std::endl;
    }

    thrust::pinned_host_vector<float> check(dst.size());
    validate_fmamat_columns<float>(src, height, width, m, c, check);

    // Print differences from validation
    printf("=============== CHECK ERRORS ============\n");
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        printf("%8.3f ",
               (double)dst[i * width + j] - (double)check[i * width + j]);
      }
      std::cout << std::endl;
    }
  }

  return 0;
}

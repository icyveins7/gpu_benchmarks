#include "blkreduction.cuh"
#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  printf("Block-wise reductions.\n");

  int batch = 32;
  int length = 1024;
  thrust::host_vector<int> src(batch * length);
  thrust::host_vector<int> maxPerBlock(batch);
  thrust::host_vector<unsigned int> argMaxPerBlock(batch);

  // Fill with some random data
  for (int i = 0; i < batch * length; i++) {
    src[i] = rand() % 100;
  }

  thrust::device_vector<int> d_src = src;
  thrust::device_vector<int> d_maxPerBlock(maxPerBlock.size());
  thrust::device_vector<unsigned int> d_argMaxPerBlock(argMaxPerBlock.size());

  // Run kernel
  int blocks = batch;
  const int threads_per_blk = 128;
  int smem = threads_per_blk * (sizeof(int) + sizeof(int));

  thrust::device_vector<int> d_debugMaxPerBlockStates(threads_per_blk * blocks);
  thrust::device_vector<unsigned int> d_debugArgMaxPerBlockStates(
      threads_per_blk * blocks);

  // simpleBlockMaxAndArgMaxKernel<<<blocks, threads_per_blk, smem>>>(
  //     d_src.data().get(), length, d_maxPerBlock.data().get(),
  //     d_argMaxPerBlock.data().get(), d_debugMaxPerBlockStates.data().get(),
  //     d_debugArgMaxPerBlockStates.data().get());

  // Cub flavour is fully templated so there's no dynamic shared mem requirement
  cubArgmax<int, threads_per_blk><<<blocks, threads_per_blk>>>(
      d_src.data().get(), length, d_maxPerBlock.data().get(),
      d_argMaxPerBlock.data().get());

  maxPerBlock = d_maxPerBlock;
  argMaxPerBlock = d_argMaxPerBlock;

  // thrust::host_vector<int> debugMaxPerBlockStates = d_debugMaxPerBlockStates;
  // thrust::host_vector<int> debugArgMaxPerBlockStates =
  //     d_debugArgMaxPerBlockStates;

  // Check
  for (int i = 0; i < batch; ++i) {
    printf("Batch %d\n", i);
    printf("Max: %u -> %d (%d directly)\n", argMaxPerBlock[i],
           src[i * length + argMaxPerBlock[i]], maxPerBlock[i]);

    // // Shared memory inspection
    // int maxInShmem = 0;
    // int argMaxInShmem = -1;
    // for (int t = 0; t < threads_per_blk; ++t) {
    //   printf("SHM [%4d]: %4d -> %4d\n", t,
    //          debugArgMaxPerBlockStates[i * threads_per_blk + t],
    //          debugMaxPerBlockStates[i * threads_per_blk + t]);
    //   if (debugMaxPerBlockStates[i * threads_per_blk + t] >= maxInShmem) {
    //     maxInShmem = debugMaxPerBlockStates[i * threads_per_blk + t];
    //     argMaxInShmem = std::max(
    //         debugArgMaxPerBlockStates[i * threads_per_blk + t],
    //         argMaxInShmem);
    //   }
    // }
    // printf("Shared memory max: %d -> %d\n", argMaxInShmem, maxInShmem);
    // printf("Verify shared memory is valid: %d -> %d\n", argMaxInShmem,
    //        src[i * length + argMaxInShmem]);

    // Direct check on host data
    int checkMax = src[i * length + 0], checkArgmax = 0;
    for (int j = 1; j < length; ++j) {
      if (src[i * length + j] > checkMax) {
        checkMax = src[i * length + j];
        checkArgmax = j;
      }
    }
    printf("Check %u -> %d (%d directly)\n", checkArgmax,
           src[i * length + checkArgmax], checkMax);
    printf("=========\n");
  }

  return 0;
}

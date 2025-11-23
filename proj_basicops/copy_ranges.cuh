#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void copy_ranges_blockwise(const T *d_src, const T *d_offsets,
                                      const T *d_lengths, T *d_dst,
                                      int num_segments) {
  for (int i = blockIdx.x; i < num_segments; i += gridDim.x) {
    for (int j = threadIdx.x; j < d_lengths[i]; j += blockDim.x) {
      d_dst[d_offsets[i] + j] = d_src[d_offsets[i] + j];
    }
  }
}

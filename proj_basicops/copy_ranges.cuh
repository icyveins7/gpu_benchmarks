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

template <typename T>
__global__ void
copy_ranges_blockwise_worksteal(const T *d_src, const T *d_offsets,
                                const T *d_lengths, T *d_dst, int num_segments,
                                unsigned long long int *d_workcounter) {
  __shared__ unsigned long long int s_segmentToWorkOn[1];

  for (int i = blockIdx.x; i < num_segments; i += gridDim.x) {
    unsigned long long int segmentToWorkOn;
    if (threadIdx.x == 0) {
      segmentToWorkOn = atomicAdd(d_workcounter, 1);
      s_segmentToWorkOn[0] = segmentToWorkOn;
    }
    __syncthreads();

    segmentToWorkOn = s_segmentToWorkOn[0];

    for (int j = threadIdx.x; j < d_lengths[segmentToWorkOn]; j += blockDim.x) {
      d_dst[d_offsets[segmentToWorkOn] + j] =
          d_src[d_offsets[segmentToWorkOn] + j];
    }
  }
}

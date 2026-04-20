#pragma once

template <typename T>
__host__ __device__ T justEnoughBlocks(const T numThreads, const T len) {
  return (len + numThreads - 1) / numThreads;
}

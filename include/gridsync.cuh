/*
This is effectively taken from the CUDA C Programming Guide:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-synchronization

Some helper functions are here so they can be easily reused.
*/

#pragma once

#include <cuda_runtime.h>

/**
 * @brief Check if the device supports cooperative launch
 *
 * @param device Device index
 * @return True if cooperative launch is supported
 */
static bool checkCooperativeLaunchSupported(int device = 0) {
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch,
                         device);

  return supportsCoopLaunch == 1;
}

/**
 * @brief Get the maximum number of blocks that should be used to fully occupy a
 * single SM. You usually want getMaxBlocksForCooperativeGrid, which multiplies
 * by the number of SMs.
 *
 * @tparam T Template for kernel, you don't have to bother with this
 * @param numThreads Number of threads per block
 * @param kernel Kernel function
 * @param device Device index
 * @return Maximum number of blocks for full occupancy
 */
template <typename T>
int getMaxBlocksPerSmForCooperativeGrid(int numThreads, T kernel,
                                        int device = 0) {
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel,
                                                numThreads, 0);

  return numBlocksPerSm;
}

/**
 * @brief Get the maximum number of blocks that should be used to fully occupy
 the device. Use this in the kernel launch parameters directly.
 *
 * @tparam T Template for kernel, you don't have to bother with this
 * @param numThreads Number of threads per block desired
 * @param kernel Kernel function
 * @param device Device index
 * @return Maximum number of blocks for full occupancy
 */
template <typename T>
int getMaxBlocksForCooperativeGrid(int numThreads, T kernel, int device = 0) {
  int numBlocksPerSm =
      getMaxBlocksPerSmForCooperativeGrid(numThreads, kernel, device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  return numBlocksPerSm * deviceProp.multiProcessorCount;
}

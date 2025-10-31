/*
 * NOTE: Currently incomplete.
 */

#pragma once
#include "cuda_runtime.h"

template <typename T> class GenericCUDAPipeline {
public:
  /**
   * @brief Copy all necessary input data from (recommended pinned) host memory
   * to device memory. You should synchronize at the end of this function after
   * you implement it.
   *
   * @param stream CUDA stream to use
   */
  void copy_input_to_device_01(cudaStream_t stream = 0) {}

  /**
   * @brief Perform the processing on the device. It is recommended that all
   * kernel launches or library calls are done asynchronously.
   *
   * @param stream CUDA stream to use
   */
  void process_device_02(cudaStream_t stream = 0) {}

  /**
   * @brief Perform the processing on the host. Example use-case: preparing
   * input for the next iteration.
   *
   * @param stream CUDA stream to use
   */
  void process_host_03(cudaStream_t stream = 0) {}

  /**
   * @brief Copy all necessary output data from device memory to (recommended
   * pinned) host memory. You should synchronize at the start of this function
   * before you implement it.
   *
   * @param stream CUDA stream to use
   */
  void copy_output_to_host_04(cudaStream_t stream = 0) {}

  void run(const size_t loops, cudaStream_t stream = 0) {
    for (size_t i = 0; i < loops; ++i) {
      // Copy current input to device
      this->copy_input_to_device_01();
      // Run current iteration on device
      this->process_device_02();
      // Run current iteration on host (in general should be async)
      this->process_host_03();
      // Copy current output to host
      this->copy_output_to_host_04();
    }
  }
};

template <typename T> class GenericCUDAPipelineDoubleBuffered {
public:
  void run(const size_t loops, cudaStream_t stream = 0) {
    // Copy the first input
    m_pipelines[0].copy_input_to_device_01(stream);

    for (size_t i = 0; i < loops; ++i) {
      size_t currIdx = i % 2;
      size_t nextIdx = (i + 1) % 2;
      // Run current input on device
      m_pipelines[currIdx].process_device_02(stream);
      // Copy next input to device
      m_pipelines[nextIdx].copy_input_to_device_01(stream);
    }
  }

protected:
  GenericCUDAPipeline<T> m_pipelines[2];
};

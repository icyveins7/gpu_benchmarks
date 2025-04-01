#pragma once

namespace containers {

/**
 * @class CudaStream
 * @brief A simple RAII container for a CUDA stream.
 *
 */
class CudaStream {
public:
  CudaStream(unsigned int flags = cudaStreamDefault);
  ~CudaStream();

  /**
   * @brief Convenience method for cudaStreamSynchronize.
   */
  void sync();

  /**
   * @brief Convenience method to get underlying stream value.
   *
   * @return m_stream, the underlying cudaStream_t.
   */
  cudaStream_t operator()();

private:
  cudaStream_t m_stream;
};

} // namespace containers

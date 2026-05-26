#pragma once

namespace containers {

/**
 * @class CudaEvent
 * @brief A simple RAII container for a CUDA event.
 *
 */
class CudaEvent {
public:
  /**
   * @brief Primary constructor for CudaEvent.
   * Acts as a wrapper around cudaEventCreate.
   *
   * @param flags Defaults to cudaEventDisableTiming for minimal overhead.
   */
  CudaEvent(unsigned int flags = cudaEventDisableTiming);
  ~CudaEvent();

  /**
   * @brief Convenience method for cudaEventRecord.
   *
   * @param stream Stream to record the event on; defaults to the default
   * stream.
   */
  void record(cudaStream_t stream = 0);

  /**
   * @brief Records the event only if the previous record has completed.
   * Queries the event first; if it returns cudaErrorNotReady, the event is
   * still in use and this method throws a std::runtime_error instead of
   * silently overwriting.
   *
   * @param stream Stream to record the event on; defaults to the default
   * stream.
   */
  void recordWithoutOverwrite(cudaStream_t stream = 0);

  /**
   * @brief Non-blocking query for event completion via cudaEventQuery.
   *
   * @return cudaSuccess if all work preceding the event has completed,
   * cudaErrorNotReady if it has not.
   */
  cudaError_t query();

  /**
   * @brief Convenience method to get underlying event value.
   *
   * @return m_event, the underlying cudaEvent_t.
   */
  cudaEvent_t &operator()();

private:
  cudaEvent_t m_event;
};

} // end namespace containers

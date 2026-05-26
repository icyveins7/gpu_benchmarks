#include <stdexcept>

#include "containers/events.cuh"

namespace containers {

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEventCreateWithFlags(&m_event, flags);
}

CudaEvent::~CudaEvent() { cudaEventDestroy(m_event); }

void CudaEvent::record(cudaStream_t stream) {
  cudaEventRecord(m_event, stream);
}

void CudaEvent::recordWithoutOverwrite(cudaStream_t stream) {
  cudaError_t status = cudaEventQuery(m_event);
  if (status == cudaErrorNotReady) {
    throw std::runtime_error(
        "CudaEvent::recordWithoutOverwrite: event is still in use.");
  }
  cudaEventRecord(m_event, stream);
}

cudaError_t CudaEvent::query() { return cudaEventQuery(m_event); }

cudaEvent_t &CudaEvent::operator()() { return m_event; }

} // namespace containers

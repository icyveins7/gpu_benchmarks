#include "containers/streams.cuh"

namespace containers {

CudaStream::CudaStream(unsigned int flags) {
  cudaStreamCreateWithFlags(&m_stream, flags);
}

CudaStream::~CudaStream() { cudaStreamDestroy(m_stream); }

void CudaStream::sync() { cudaStreamSynchronize(m_stream); }

cudaStream_t &CudaStream::operator()() { return m_stream; }

void CudaStream::wait(cudaEvent_t &event) {
  cudaStreamWaitEvent(m_stream, event, 0);
}

} // namespace containers

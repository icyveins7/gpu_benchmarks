#include "containers.cuh"

namespace containers {

CudaStream::CudaStream(unsigned int flags) {
  cudaStreamCreateWithFlags(&m_stream, flags);
}

CudaStream::~CudaStream() { cudaStreamDestroy(m_stream); }

void CudaStream::sync() { cudaStreamSynchronize(m_stream); }

cudaStream_t CudaStream::operator()() { return m_stream; }

} // namespace containers

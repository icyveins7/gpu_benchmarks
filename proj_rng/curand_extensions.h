#include <curand.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

// Kernel function to initialize cuRAND states
// This grid strides over the entire length to initialize based on the
// index of the data, not the thread
static __global__ void setup_kernel(curandState *state, unsigned long seed, const size_t length) {
  for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < length; t += blockDim.x * gridDim.x) {
    curand_init(seed, t, 0, &state[t]);
  }
}

// Kernel function to generate random numbers
// Same grid stride as the setup_kernel
static __global__ void rand_kernel(float *out, curandState *state, const size_t length) {
  for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < length; t += blockDim.x * gridDim.x) {
    out[t] = curand_uniform(&state[t]);
  }
}

template <typename T>
class CuRandRNG
{
public:
  CuRandRNG(const unsigned long seed, const size_t size) : m_seed(seed) {
    setup(size);
  }

  void rand(thrust::device_vector<T> &out);

protected:
  const unsigned long m_seed;
  thrust::device_vector<curandState> m_rngstates;

  void setup(const size_t size){
    m_rngstates.resize(size);
    const int threadsPerBlk = 256;
    const int n_blocks = (size + threadsPerBlk - 1) / threadsPerBlk;
    setup_kernel<<<n_blocks, threadsPerBlk>>>(thrust::raw_pointer_cast(m_rngstates.data()), m_seed, size);
  }

};

// specialisation for float for now only
template<>
inline void CuRandRNG<float>::rand(thrust::device_vector<float> &out) {
  if (out.size() != m_rngstates.size()) {
    throw std::runtime_error("CuRandRNG::rand: out and rngstates have different sizes");
  }
  const int threadsPerBlk = 256;
  const int n_blocks = (out.size() + threadsPerBlk - 1) / threadsPerBlk;
  rand_kernel<<<n_blocks, threadsPerBlk>>>(
    thrust::raw_pointer_cast(out.data()),
    thrust::raw_pointer_cast(m_rngstates.data()),
    out.size()
  );
}

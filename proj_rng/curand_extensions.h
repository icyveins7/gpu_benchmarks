#include <curand.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

// Kernel function to initialize cuRAND states
// This grid strides over the entire length to initialize based on the
// index of the data, not the thread
static __global__ void setup_kernel(curandState *state, const unsigned long long seed,
                                    const size_t length, const unsigned long long offset) {
  for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < length; t += blockDim.x * gridDim.x) {
    curand_init(seed, t, offset, &state[t]);
  }
}

static __global__ void skipahead_kernel(unsigned long long n, curandState *state, const size_t length) {
  for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < length; t += blockDim.x * gridDim.x) {
    skipahead(n, &state[t]);
  }
}

// Kernel function to generate random numbers
// Same grid stride as the setup_kernel
static __global__ void rand_kernel(float *out, curandState *state, const size_t length) {
  for (int t = threadIdx.x + blockIdx.x * blockDim.x; t < length; t += blockDim.x * gridDim.x) {
    out[t] = curand_uniform(&state[t]);
  }
}

/**
 * @brief Generates random numbers in a patch at the start (top-left) of each image,
 *        in a batch of images.
 *
 *        NOTE: to make things simple, the number of initialized curand states is expected
 *        to be the same total size as (batch size * patch height * patch width). This ensures that
 *        for a given new batch, every state only needs to be forwarded once and only once, rather
 *        than some complicated combination.
 *
 * @example
 *
 *   X X - -
 *   X X - -
 *   - - - -
 *   - - - -
 *
 *   This would have a patch width/height of 2, and output width/height of 4.
 *   There would be a few of these in 1 batch.
 *
 * @param out Output array
 * @param state cuRAND states array
 * @param batch Number of patches in the batch
 * @param oWidth Output image width
 * @param oHeight Output image height
 * @param patchWidth Patch width within the image
 * @param patchHeight Patch height within the image
 * @return 
 */
static __global__ void randPatchInBatch_kernel(
  float *out, curandState *state, const unsigned int batch,
  const unsigned int oWidth, const unsigned int oHeight,
  const unsigned int patchWidth, const unsigned int patchHeight)
{
  // Standard grid-stride over the patches
  unsigned int elemIdx, patchIdx, row, col, outIdx;
  for (unsigned int t = threadIdx.x + blockIdx.x * blockDim.x; t < batch * patchWidth * patchHeight;
       t += blockDim.x * gridDim.x)
  {
    // Patch index i.e. which patch inside the batch
    patchIdx = t / (patchWidth * patchHeight);
    // Element index i.e. which element inside the patch
    elemIdx = t % (patchWidth * patchHeight);
    // Unroll patch index into row and column (assuming trivial contiguous memory layout)
    col = elemIdx % patchWidth;
    row = elemIdx / patchWidth;

    // Now calculate the output index (inside the larger image)
    outIdx = patchIdx * oWidth * oHeight + row * oWidth + col;

    // Otherwise, write to the output
    out[outIdx] = curand_uniform(&state[t]);
  }
}


template <typename T>
class CuRandRNG
{
public:
  /**
   * @brief Constructs a simple CuRandRNG object which initializes a specified number of states.
   *        It is assumed that the number of written output elements per invocation is equal
   *        to the number of states; this is to facilitate book-keeping and easier fast-forwarding/
   *        skipahead to track the current state in the sequence.
   *
   * @param seed Initial seed value
   * @param size Number of curand states, equivalent to number of output elements later
   * @param offset Initial offset; this should be equivalent to number of rand() invocations
   */
  CuRandRNG(const unsigned long long seed, const size_t size, const unsigned long long offset = 0)
    : m_seed(seed), m_offset(offset) {
    setup(size);
  }

  /**
   * @brief Gets the current offset value, accumulated from rand() or skipahead() calls.
   *
   * @return Current offset value
   */
  unsigned long long getOffset() const { return m_offset; }

  /**
   * @brief Skips-ahead an equivalent offset value for each state.
   *        Note that the offset is correlated with the compute time i.e.
   *        larger offset means it will take longer.
   *
   * @param n Number of states to skip ahead in the RNG sequence, for each state
   */
  void skipahead(const unsigned long long n){
    const int threadsPerBlk = 256;
    const int n_blocks = (m_rngstates.size() + threadsPerBlk - 1) / threadsPerBlk;
    skipahead_kernel<<<n_blocks, threadsPerBlk>>>(n, thrust::raw_pointer_cast(m_rngstates.data()), m_rngstates.size());
    // increment offset to keep track
    m_offset += n;
  }


  /**
   * @brief Main run-time method to generate random numbers.
   *        Expects output elements to be same length as the internal states.
   *
   * @param out Output vector
   */
  void rand(thrust::device_vector<T> &out);

protected:
  unsigned long long m_seed;
  unsigned long long m_offset;
  thrust::device_vector<curandState> m_rngstates;

  void setup(const size_t size){
    m_rngstates.resize(size);
    const int threadsPerBlk = 256;
    const int n_blocks = (size + threadsPerBlk - 1) / threadsPerBlk;
    setup_kernel<<<n_blocks, threadsPerBlk>>>(thrust::raw_pointer_cast(m_rngstates.data()), m_seed, size, m_offset);
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

  // increment internal offset to keep track
  m_offset++;
}

// ================================================================================
template <typename T>
class CuRandRNGPatchInBatch : public CuRandRNG<T>
{
public:
  CuRandRNGPatchInBatch(const unsigned long long seed, const unsigned int batch,
                        const unsigned int patchWidth, const unsigned int patchHeight,
                        const unsigned long long offset = 0)
    : CuRandRNG<T>(seed, batch * patchWidth * patchHeight, offset),
      m_batch(batch),
      m_patchWidth(patchWidth), m_patchHeight(patchHeight)
  {}

  void rand(thrust::device_vector<T> &out, const unsigned int oWidth, const unsigned int oHeight); 

private:
  const unsigned int m_batch;
  const unsigned int m_patchWidth;
  const unsigned int m_patchHeight;

  // 'remove' original rand() implementation
  using CuRandRNG<T>::rand;
};

// specialisation
template <>
inline void CuRandRNGPatchInBatch<float>::rand(
  thrust::device_vector<float> &out, const unsigned int oWidth, const unsigned int oHeight) 
{
  // Ensure the vector matches the expected size
  if (out.size() != m_batch * oWidth * oHeight) {
    throw std::runtime_error("CuRandRNGPatchInBatch::rand: out and rngstates have different sizes");
  }
  // Ensure output dimensions are larger than the patch dimensions
  if (oWidth < m_patchWidth || oHeight < m_patchHeight) {
    throw std::runtime_error("CuRandRNGPatchInBatch::rand: output dimensions too small, must be larger than patch dimensions");
  }
  const int threadsPerBlk = 256;
  const int n_blocks = m_rngstates.size() / threadsPerBlk + 1;
  randPatchInBatch_kernel<<<n_blocks, threadsPerBlk>>>(
    thrust::raw_pointer_cast(out.data()),
    thrust::raw_pointer_cast(this->m_rngstates.data()),
    m_batch, oWidth, oHeight,
    m_patchWidth, m_patchHeight
  );

  // increment internal offset to keep track
  this->m_offset++;
}

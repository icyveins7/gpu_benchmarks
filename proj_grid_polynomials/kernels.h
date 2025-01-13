#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <random>

#include <stdexcept>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAX_COEFFS 32 // this is only used in the constant memory implementation

/**
 * @brief Device-side function that computes the polynomial for a given
 *        set of coefficients. Assumes a single-thread approach.
 *
 * @tparam T Type of coefficients, input and output
 * @param coeffs Pointer to coefficients
 * @param numCoeffs Number of coefficients
 * @param in Pointer to input
 * @param out Pointer to output
 * @return 
 */
template <typename T>
__device__ void computePolynomialForValue(const T *const coeffs,
                                          const int numCoeffs, const T in,
                                          T &out) {
  // Add 0-order term
  out = coeffs[0];
  // Start at power 1
  T inp = in;
  for (size_t i = 1; i < numCoeffs; ++i) {
    // Add next term
    out += inp * coeffs[i];
    // Increment power
    inp = inp * in;
  }
}



/**
 * @brief Base class to hold polynomial coefficients.
 *        Also contains empty runner methods for host and device;
 *        these are to be re-implemented in derived classes based
 *        on custom kernels and their intended use-cases.
 *
 * @tparam T Type of coefficients, input and output.
 * @param numCoeffs Total number of coefficients
 * @return 
 */
template <typename T>
class GridPolynom
{
public:
  /**
   * @brief Constructor to randomize coefficients
   *
   * @param numCoeffs Number of coefficients to use
   */
  GridPolynom(size_t numCoeffs){
    // Randomize number of coeffs on host
    m_h_coeffs.resize(numCoeffs);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (T &coeff : m_h_coeffs) {
      coeff = dis(gen);
    }

    // Transfer immediately to device
    // No need for async shenanigans
    m_d_coeffs = m_h_coeffs;
  }

  /**
   * @brief Constructor to simply copy coefficients
   *
   * @param coeffs Coefficients to use
   */
  GridPolynom(const thrust::host_vector<T>& coeffs)
    : m_h_coeffs(coeffs) 
  {
    // Transfer immediately to device
    // No need for async shenanigans
    m_d_coeffs = m_h_coeffs;
  };

  // Primarily for checking
  inline const thrust::host_vector<T>& h_coeffs() const { return m_h_coeffs; }
  inline const thrust::device_vector<T>& d_coeffs() const { return m_d_coeffs; }

  // Placeholders for actual computations to do in child classes
  void h_run(const thrust::host_vector<T>& h_in, thrust::host_vector<T>& h_out) {};
  void d_run(const thrust::device_vector<T>& d_in, thrust::device_vector<T>& d_out) {};

protected:
  // All coefficients should be order 0 to order N
  thrust::host_vector<T> m_h_coeffs;
  thrust::device_vector<T> m_d_coeffs;
};

// =============================================================================

/**
 * @brief A simple kernel that should compute the polynomial for every input,
 *        with 1 input per thread. This will read the coefficients from global
 *        memory.
 *
 * @tparam T Input/output/coefficients type (either float or double)
 * @param d_coeffs Device pointer to coefficients
 * @param numCoeffs Number of coefficients
 * @param in Input array pointer
 * @param in_length Number of elements in input array
 * @param out Output array pointer
 * @return 
 */
template <typename T>
__global__ void
naiveGridStridePolynomial(const T *const d_coeffs, const size_t numCoeffs,
                          const T *const in, const size_t in_length, T *out) {
  // Simply execute 1 thread -> 1 value
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Assume enough threads spawned to cover, no need to stride
  T tmp;
  computePolynomialForValue(d_coeffs, numCoeffs, in[idx], tmp);
  // Coalesced global write
  out[idx] = tmp;
}

template <typename T>
class NaiveGridPolynom : public GridPolynom<T>
{
public:
  NaiveGridPolynom(size_t numCoeffs) : GridPolynom<T>(numCoeffs) {};
  NaiveGridPolynom(const thrust::host_vector<T>& coeffs) : GridPolynom<T>(coeffs) {};

  void h_run(const thrust::host_vector<T>& h_in, thrust::host_vector<T>& h_out) {
    h_out.resize(h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i)
    {
      h_out[i] = h_in[i]; // TODO: complete this

    }
  };

  void d_run(const thrust::device_vector<T>& d_in, thrust::device_vector<T>& d_out) {
    // Extract raw device pointers for kernel
    int THREADS_PER_BLK = 128;
    int numBlks = static_cast<int>(d_in.size() / THREADS_PER_BLK) + 1;

    naiveGridStridePolynomial<<<numBlks, THREADS_PER_BLK>>>(
      thrust::raw_pointer_cast(this->m_d_coeffs.data()),
      this->m_d_coeffs.size(),
      thrust::raw_pointer_cast(d_in.data()),
      d_in.size(),
      thrust::raw_pointer_cast(d_out.data())
    );
  };

};

// =============================================================================

/**
 * @brief Identical to naiveGridStridePolynomial, but stores coefficients in shared
 *        memory first. Measured to have no noticeable difference (and may be slightly
 *        slower in fact).
 *
 * @tparam T Input/output/coefficients type (either float or double)
 * @param d_coeffs Device pointer to coefficients
 * @param numCoeffs Number of coefficients
 * @param in Input array pointer
 * @param in_length Number of elements in input array
 * @param out Output array pointer
 * @return 
 */
template <typename T>
__global__ void
sharedCoeffsGridStridePolynomial(const T *const d_coeffs, const size_t numCoeffs,
                                 const T *const in, const size_t in_length, T *out) {
  // Allocate shared memory for the coefficients
  extern __shared__ float s[];
  float *s_coeffs = s;

  // Read coeffs into shared memory
  for (int t = threadIdx.x; t < numCoeffs; ++t)
    s_coeffs[t] = d_coeffs[t];

  __syncthreads();

  // Compute polynomial
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T tmp;
  computePolynomialForValue(s_coeffs, numCoeffs, in[idx], tmp);
  // Coalesced global write
  out[idx] = tmp;
}

template <typename T>
class SharedCoeffGridPolynom : public GridPolynom<T>
{
public:
  SharedCoeffGridPolynom(size_t numCoeffs) : GridPolynom<T>(numCoeffs) {};
  SharedCoeffGridPolynom(const thrust::host_vector<T>& coeffs) : GridPolynom<T>(coeffs) {};

  void d_run(const thrust::device_vector<T>& d_in, thrust::device_vector<T>& d_out) {
    // Extract raw device pointers for kernel
    int THREADS_PER_BLK = 128;
    int numBlks = static_cast<int>(d_in.size() / THREADS_PER_BLK) + 1;

    sharedCoeffsGridStridePolynomial<<<numBlks, THREADS_PER_BLK, sizeof(T)*this->m_d_coeffs.size()>>>(
      thrust::raw_pointer_cast(this->m_d_coeffs.data()),
      this->m_d_coeffs.size(),
      thrust::raw_pointer_cast(d_in.data()),
      d_in.size(),
      thrust::raw_pointer_cast(d_out.data())
    );
  };
};

// =============================================================================

template <typename T>
struct constCoeffsStruct{
  T c_coeffs[MAX_COEFFS];
};

// We use double since it's enough for both double and float
static __constant__ double c_coeffs[MAX_COEFFS];

/**
 * @brief Identical to naiveGridStridePolynomial, but stores coefficients in constant
 *        memory instead. Since the number of coefficients is much less than the
 *        maximum parameter size limit (4096 bytes, and 32764 bytes in CUDA 12.1!),
 *        we can just pass the entire thing as a struct directly.
 *
 *        https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/
 *
 *        Doing this automatically caches it into constant memory (as do all
 *        kernel arguments).
 *
 * @tparam T Input/output/coefficients type (either float or double)
 * @param d_coeffs Device pointer to coefficients
 * @param numCoeffs Number of coefficients
 * @param in Input array pointer
 * @param in_length Number of elements in input array
 * @param out Output array pointer
 * @return 
 */
template <typename T>
__global__ void
// constantCoeffsGridStridePolynomial(const constCoeffsStruct<T> coeffStruct, const size_t numCoeffs,
//                                    const T *const in, const size_t in_length, T *out) {
constantCoeffsGridStridePolynomial(const size_t numCoeffs,
                                   const T *const in, const size_t in_length, T *out) {
  // Compute polynomial
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T tmp;
  computePolynomialForValue(reinterpret_cast<const T*>(c_coeffs), numCoeffs, in[idx], tmp);
  // Coalesced global write
  out[idx] = tmp;
}

template <typename T>
class ConstantCoeffGridPolynom : public GridPolynom<T>
{
public:
  ConstantCoeffGridPolynom(size_t numCoeffs) : GridPolynom<T>(numCoeffs) {
    // Check if coefficients exceed constant mem allocation
    if (this->m_h_coeffs.size() > MAX_COEFFS)
      throw std::invalid_argument("MAX_COEFFS exceeded for constant memory!");

    // We copy additionally to the struct
    for (int i = 0; i < this->m_h_coeffs.size(); ++i)
      this->m_coeffStruct.c_coeffs[i] = this->m_h_coeffs[i];

    // And copy the symbol manually too
    cudaMemcpyToSymbol(c_coeffs,
                       thrust::raw_pointer_cast(this->m_h_coeffs.data()),
                       sizeof(T)*this->m_h_coeffs.size());

  };
  ConstantCoeffGridPolynom(const thrust::host_vector<T>& coeffs) : GridPolynom<T>(coeffs) {
    // Check if coefficients exceed constant mem allocation
    if (this->m_h_coeffs.size() > MAX_COEFFS)
      throw std::invalid_argument("MAX_COEFFS exceeded for constant memory!");

    // We copy additionally to the struct
    for (int i = 0; i < this->m_h_coeffs.size(); ++i)
      this->m_coeffStruct.c_coeffs[i] = this->m_h_coeffs[i];

    // And copy the symbol manually too
    cudaMemcpyToSymbol(c_coeffs,
                       thrust::raw_pointer_cast(this->m_h_coeffs.data()),
                       sizeof(T)*this->m_h_coeffs.size());
  };

  void h_run(const thrust::host_vector<T>& h_in, thrust::host_vector<T>& h_out) {
    h_out.resize(h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i)
    {
      h_out[i] = h_in[i]; // TODO: complete this

    }
  };

  void d_run(const thrust::device_vector<T>& d_in, thrust::device_vector<T>& d_out) {
    // Extract raw device pointers for kernel
    int THREADS_PER_BLK = 128;
    int numBlks = static_cast<int>(d_in.size() / THREADS_PER_BLK) + 1;

    constantCoeffsGridStridePolynomial<<<numBlks, THREADS_PER_BLK>>>(
      // this->m_coeffStruct,
      this->m_h_coeffs.size(),
      thrust::raw_pointer_cast(d_in.data()),
      d_in.size(),
      thrust::raw_pointer_cast(d_out.data())
    );
  };

private:
  constCoeffsStruct<T> m_coeffStruct;
};


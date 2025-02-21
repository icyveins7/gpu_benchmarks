/*
Provide some convenient wrappers to handle the plans.
*/

#include <cufft.h>

#include <stdexcept>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <array>
#include <complex>
#include <stdexcept>
#include <string>

// Helper method for error checking
void CUFFT_NO_ERROR(cufftResult t) {
  if (t != cufftResult::CUFFT_SUCCESS)
    throw std::runtime_error("CUFFT failed with code " + std::to_string(t));
}

/*
==========================================
1D Wrapper
==========================================
*/
template <cufftType T> class cuFFTWrapper_1D {
public:
  cuFFTWrapper_1D(size_t fftSize, size_t batchSize)
      : m_fftSize(fftSize), m_batchSize(batchSize) {
    createPlan();
  }
  // Make non-copyable (and non-movable by extrapolation)
  cuFFTWrapper_1D(const cuFFTWrapper_1D &) = delete;
  cuFFTWrapper_1D &operator=(const cuFFTWrapper_1D &) = delete;
  // Defining copy ctor/assignment stops compiler from emitting default move
  // semantics

  ~cuFFTWrapper_1D() { destroyPlan(); }

  inline size_t fftSize() const { return m_fftSize; }
  inline size_t batchSize() const { return m_batchSize; }
  // Number of complex output values per row in the batch
  inline size_t outputPerRow() const { return m_fftSize / 2 + 1; }
  // Use this value to allocate an output array of sufficient size
  inline size_t outputSize() const { return outputPerRow() * m_batchSize; }

  // Primary runtime method
  template <typename U, typename V>
  void exec(const thrust::device_vector<U> &input,
            thrust::device_vector<V> &output);

private:
  size_t m_fftSize = 0;
  size_t m_batchSize = 0;

  cufftHandle plan;

  void createPlan() {
    CUFFT_NO_ERROR(cufftCreate(&plan));
    CUFFT_NO_ERROR(cufftPlan1d(&plan, static_cast<int>(m_fftSize), T,
                               static_cast<int>(m_batchSize)));
  }

  void destroyPlan() { CUFFT_NO_ERROR(cufftDestroy(plan)); }

  // Checks whether the input size is sufficient for the plan execution
  void validateInputSize(size_t sz) {
    if (sz < m_fftSize * m_batchSize)
      throw std::invalid_argument("Input size is too small!");
  }

  // Checks whether the output size is sufficient for the plan execution
  void validateOutputSize(size_t sz) {
    if (sz < outputSize())
      throw std::invalid_argument("Output size is too small!");
  }
};

/*
Flavours of exec()
*/

// R2C
template <>
template <>
inline void cuFFTWrapper_1D<CUFFT_R2C>::exec(
    const thrust::device_vector<float> &input,
    thrust::device_vector<std::complex<float>> &output) {
  CUFFT_NO_ERROR(cufftExecR2C(
      plan,
      const_cast<cufftReal *>(
          thrust::reinterpret_pointer_cast<const cufftReal *>(input.data())),
      thrust::reinterpret_pointer_cast<cufftComplex *>(output.data())));
}

// C2R
template <>
template <>
inline void cuFFTWrapper_1D<CUFFT_C2R>::exec(
    const thrust::device_vector<std::complex<float>> &input,
    thrust::device_vector<float> &output) {
  CUFFT_NO_ERROR(cufftExecC2R(
      plan,
      const_cast<cufftComplex *>(
          thrust::reinterpret_pointer_cast<const cufftComplex *>(input.data())),
      thrust::reinterpret_pointer_cast<cufftReal *>(output.data())));
}

/*
==========================================
2D Wrapper
==========================================
*/
template <cufftType T> class cuFFTWrapper_2D {
public:
  // Note that the current implementation assumes contiguous storage
  // i.e. no stride/dist/embed parameters are relevant
  // Note that fftSize is given as (rows, columns)
  cuFFTWrapper_2D(std::array<int, 2> fftSize, size_t batchSize)
      : m_fftSize(fftSize), m_batchSize(batchSize) {
    createPlan();
  }
  // Make non-copyable (and non-movable by extrapolation)
  cuFFTWrapper_2D(const cuFFTWrapper_2D &) = delete;
  cuFFTWrapper_2D &operator=(const cuFFTWrapper_2D &) = delete;
  // Defining copy ctor/assignment stops compiler from emitting default move
  // semantics

  ~cuFFTWrapper_2D() { destroyPlan(); }

  inline std::array<int, 2> fftSize() const { return m_fftSize; }
  inline size_t batchSize() const { return m_batchSize; }
  // Number of complex output values (no. of columns) per row in 1 image in the
  // batch. The number of rows does not change
  inline size_t outputPerRow() const { return m_fftSize[1] / 2 + 1; }
  // Use this value to allocate an output array of sufficient size
  inline size_t outputSize() const {
    return m_fftSize[0] * outputPerRow() * m_batchSize;
  }

  // Primary runtime method
  template <typename U, typename V>
  void exec(const thrust::device_vector<U> &input,
            thrust::device_vector<V> &output);

private:
  std::array<int, 2> m_fftSize = {0, 0};
  size_t m_batchSize = 0;

  cufftHandle plan;

  void createPlan() {
    CUFFT_NO_ERROR(cufftPlanMany(&plan, m_fftSize.size(), m_fftSize.data(),
                                 nullptr, 1, 0, nullptr, 1, 0, T,
                                 static_cast<int>(m_batchSize)));
  }

  void destroyPlan() { CUFFT_NO_ERROR(cufftDestroy(plan)); }

  // Checks whether the input size is sufficient for the plan execution
  void validateInputSize(size_t sz) {
    if (sz < m_fftSize[0] * m_fftSize[1] * m_batchSize)
      throw std::invalid_argument("Input size is too small!");
  }

  // Checks whether the output size is sufficient for the plan execution
  void validateOutputSize(size_t sz) {
    if (sz < outputSize())
      throw std::invalid_argument("Output size is too small!");
  }
};

/*
Flavours of exec(), literally exactly the same as the 1D case..
*/

// R2C
template <>
template <>
inline void cuFFTWrapper_2D<CUFFT_R2C>::exec(
    const thrust::device_vector<float> &input,
    thrust::device_vector<std::complex<float>> &output) {
  CUFFT_NO_ERROR(cufftExecR2C(
      plan,
      const_cast<cufftReal *>(
          thrust::reinterpret_pointer_cast<const cufftReal *>(input.data())),
      thrust::reinterpret_pointer_cast<cufftComplex *>(output.data())));
}

// C2R
template <>
template <>
inline void cuFFTWrapper_2D<CUFFT_C2R>::exec(
    const thrust::device_vector<std::complex<float>> &input,
    thrust::device_vector<float> &output) {
  CUFFT_NO_ERROR(cufftExecC2R(
      plan,
      const_cast<cufftComplex *>(
          thrust::reinterpret_pointer_cast<const cufftComplex *>(input.data())),
      thrust::reinterpret_pointer_cast<cufftReal *>(output.data())));
}

// C2C
template <>
template <>
inline void cuFFTWrapper_2D<CUFFT_C2C>::exec(
    const thrust::device_vector<std::complex<float>> &input,
    thrust::device_vector<std::complex<float>> &output) {
  CUFFT_NO_ERROR(cufftExecC2C(
      plan,
      const_cast<cufftComplex *>(
          thrust::reinterpret_pointer_cast<const cufftComplex *>(input.data())),
      thrust::reinterpret_pointer_cast<cufftComplex *>(output.data()),
      CUFFT_FORWARD));
}

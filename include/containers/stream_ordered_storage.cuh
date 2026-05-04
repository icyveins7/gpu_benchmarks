#pragma once

#include <stdexcept>
#include <string>

namespace containers {

/**
 * @brief Light RAII wrapper around cudaMallocAsync. All stream-specific
 * behaviour still applies; the wrapper simply helps to run cudaFreeAsync on the
 * pointer using the same stream at the end of the scope.
 *
 * @detail If you do not understand the pros/cons of stream-ordered allocation,
 * please do not use this blindly.
 * The use-case of cudaMallocAsync is usually to reuse scratch memory that has
 * been cudaFreeAsync-ed previously. This incurs almost no runtime cost when the
 * memory previously allocated and freed is sufficient to service the current
 * request. As such, resizing the array is not really the mindset to have for
 * this class, and is intentionally left out for now.
 *
 * @example A common use-case is to actively scope a kernel launch and simply
 construct this at the beginning of the scope for scratch workspace.
 ```
  {
    StreamOrderedDeviceStorage<float> storage(1024, stream);
    kernel<<<1, 1, 0, stream>>>(storage.data());
    // storage will automatically be freed via cudaFreeAsync here
  }
  {
    // this 2nd allocation incurs almost no runtime cost,
    // since the stream has not been synchronized
    StreamOrderedDeviceStorage<float> storage2(1024, stream);
    kernel<<<1, 1, 0, stream>>>(storage2.data());
    // storage will automatically be freed via cudaFreeAsync here
  }
 ```
 *
 * @tparam T Type of data
 * @return StreamOrderedDeviceStorage
 */
template <typename T> class StreamOrderedDeviceStorage {
public:
  /**
   * @brief Default constructor. Sets everything to 0 / null.
   */
  StreamOrderedDeviceStorage()
      : m_capacity(0), m_size(0), m_data(nullptr), m_stream(0) {

        };

  /**
   * @brief Primary constructor for StreamOrderedDeviceStorage.
   * Acts as a wrapper around cudaMallocAsync.
   *
   * @param size Number of elements
   * @param stream Stream to use for allocation
   */
  explicit StreamOrderedDeviceStorage(const size_t size,
                                      const cudaStream_t stream)
      : m_capacity(size), m_size(size), m_data(nullptr), m_stream(stream) {
    alloc();
  }

  ~StreamOrderedDeviceStorage() {
#ifndef NDEBUG
    printf("StreamOrderedDeviceStorage: cudaFreeAsync of %zu bytes\n", m_size);
#endif
    cudaError_t err = cudaFreeAsync(m_data, m_stream);
    if (err != cudaSuccess) {
      // Since dtors are noexcept, we simply print the error
      // The only sensible error will come when either the stream or the
      // memory has been invalidated externally before this,
      printf("cudaFreeAsync failed with error: %s\n", cudaGetErrorString(err));
    }
  }

  // No copy semantics whatsoever
  StreamOrderedDeviceStorage(const StreamOrderedDeviceStorage &) = delete;
  StreamOrderedDeviceStorage &
  operator=(const StreamOrderedDeviceStorage &) = delete;

  // Move-only
  StreamOrderedDeviceStorage(StreamOrderedDeviceStorage &&other) {
    m_capacity = other.m_capacity;
    m_size = other.m_size;
    m_data = other.m_data;
    m_stream = other.m_stream;
    other.m_data = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;
  }
  StreamOrderedDeviceStorage &operator=(StreamOrderedDeviceStorage &&other) {
    m_capacity = other.m_capacity;
    m_size = other.m_size;
    m_data = other.m_data;
    m_stream = other.m_stream;
    other.m_data = nullptr;
    other.m_size = 0;
    other.m_capacity = 0;
    return *this;
  };

  const T *data() const { return m_data; }
  T *data() { return m_data; }
  size_t size() const { return m_size; }
  size_t capacity() const { return m_capacity; }
  cudaStream_t stream() const { return m_stream; }

  /**
   * @brief Run-time initialization. Useful when this vector is used as a member
   * of another struct, but cannot be list-initialized.
   *
   * @param stream Valid non-default stream.
   */
  void initialize(size_t size, cudaStream_t stream) {
    if (m_data != nullptr) {
      throw std::runtime_error("Already initialized");
    }
    if (stream == 0)
      throw std::runtime_error("Must not be a default stream");
    m_stream = stream;
    m_size = size;
    m_capacity = size;
    alloc();
  }

protected:
  size_t m_capacity;
  size_t m_size;
  T *m_data;
  cudaStream_t m_stream;

  void alloc() {
#ifndef NDEBUG
    printf("StreamOrderedDeviceStorage: cudaMallocAsync of %zu bytes\n",
           m_size);
#endif
    cudaError_t err = cudaMallocAsync(&m_data, sizeof(T) * m_size, m_stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "cudaMallocAsync failed with size " +
          std::to_string(m_size * sizeof(T)) + " and stream " +
          std::to_string(reinterpret_cast<uint64_t>(m_stream)));
    }
  }
};

} // namespace containers

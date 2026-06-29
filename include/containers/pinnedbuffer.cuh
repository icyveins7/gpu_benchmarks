#pragma once

#include <cstring>
#include <stdexcept>

namespace containers {

/**
 * @brief Container for pinned host memory. Alternative to
 * thrust::pinned_host_vector (see pinnedalloc.cuh).
 *
 * @detail Important difference from thrust is that this performs no zero-ing;
 * thrust will launch a kernel that zeros the memory right after via a mapped
 * host pointer i.e. kernel works on CPU memory (which is very slow), also
 * causing a synchronization.
 *
 * TODO: currently does not have copy/move semantics.
 *
 * @tparam T Data type
 */
template <typename T> class PinnedHostBuffer {
public:
  PinnedHostBuffer() : m_data{nullptr}, m_size(0), m_capacity(0) {}
  PinnedHostBuffer(size_t size) {
    m_size = size;
    m_capacity = size;
    cudaError_t err = cudaMallocHost(&m_data, size * sizeof(T));
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMallocHost failed with size " +
                               std::to_string(size * sizeof(T)));
    }
  }
  ~PinnedHostBuffer() {
    cudaError_t err = cudaFreeHost(m_data);
    if (err != cudaSuccess) {
      printf("cudaFreeHost failed with error: %s\n", cudaGetErrorString(err));
    }
  }

  void resize(size_t size) {
    if (size > m_capacity) {
      // Reallocate
      realloc(size);
    }
    m_size = size;
  }

  T* data() { return m_data; }
  const T* data() const { return m_data; }
  size_t size() const { return m_size; }
  size_t capacity() const { return m_capacity; }

  T& at(size_t i) {
    if (i >= m_size)
      throw std::out_of_range("PinnedHostBuffer::at: index " +
                              std::to_string(i) + " out of range " +
                              std::to_string(m_size));
    return m_data[i];
  }
  const T& at(size_t i) const {
    if (i >= m_size)
      throw std::out_of_range("PinnedHostBuffer::at: index " +
                              std::to_string(i) + " out of range " +
                              std::to_string(m_size));
    return m_data[i];
  }

protected:
  T* m_data;
  size_t m_size;
  size_t m_capacity;

  void realloc(size_t size) {
    T* newdata = nullptr;
    cudaError_t err = cudaMallocHost(&newdata, size * sizeof(T));
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMallocHost failed with size " +
                               std::to_string(size * sizeof(T)));
    }
    m_capacity = size;

    // Copy over
    if (m_data != nullptr && m_size > 0)
      std::memcpy(newdata, m_data, m_size * sizeof(T));

    // Free old
    cudaFreeHost(m_data);

    // Set new
    m_data = newdata;
  }
};

} // namespace containers

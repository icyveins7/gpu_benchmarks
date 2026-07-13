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
  /**
   * @brief Primary constructor. Allows setting of allocation flags, unlike
   * thrust vectors.
   *
   * @param size Number of elements
   * @param flags Allocation flags, defaults to cudaHostAllocDefault. Other
   * useful flags are cudaHostAllocPortable and cudaHostAllocMapped. Note that
   * current observations show CUDA assigning cudaHostAllocMapped regardless of
   * what was requested; this means that setting cudaHostAllocPortable is
   * actually equivalent to setting cudaHostAllocPortable | cudaHostAllocMapped.
   */
  PinnedHostBuffer(size_t size, unsigned int flags = cudaHostAllocDefault) {
    m_size = size;
    m_capacity = size;
    cudaError_t err = cudaHostAlloc(&m_data, size * sizeof(T), flags);
    if (err != cudaSuccess) {
      printf("cudaHostAlloc failed with error: %s\n", cudaGetErrorString(err));
      throw std::runtime_error("cudaHostAlloc failed with size " +
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
    if (size > m_capacity)
      realloc(size, flags());
    m_size = size;
  }

  /**
   * @brief Same as resize, but allows setting/change of allocation flags.
   * Useful if resizing from the default constructor.
   *
   * Forces reallocation if the flags differ from the current allocation or if
   * the buffer has not been allocated yet.
   *
   * @param size New number of elements
   * @param new_flags New allocation flags
   */
  void resize(size_t size, unsigned int new_flags) {
    if (m_data == nullptr || flags() != new_flags || size > m_capacity)
      realloc(size, new_flags);
    m_size = size;
  }

  /**
   * @brief Returns the allocation flags of the current buffer, as reported by
   * cudaHostGetFlags. Returns cudaHostAllocDefault if not yet allocated.
   *
   * @detail cudaHostGetFlags reports page-level flags, not allocation-level
   * flags. The first allocation on a page fixes that page's GPU registration,
   * and all subsequent allocations on the same page inherit those flags
   * regardless of what was requested. This means:
   *   - A default allocation followed by a portable one on the same page will
   *     report flags=2 (mapped only) for the portable allocation.
   *   - A portable allocation followed by a default one on the same page will
   *     report flags=3 (portable | mapped) for the default allocation.
   * To reliably obtain portable memory, either use cudaHostAllocPortable for
   * all pinned allocations, or allocate portably before any default
   * allocations so pages are registered as portable from the outset.
   */
  unsigned int flags() const {
    if (m_data == nullptr)
      return cudaHostAllocDefault;
    unsigned int f = 0;
    cudaHostGetFlags(&f, m_data);
    return f;
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

  void realloc(size_t size, unsigned int flags) {
    T* newdata = nullptr;
    cudaError_t err = cudaHostAlloc(&newdata, size * sizeof(T), flags);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaHostAlloc failed with size " +
                               std::to_string(size * sizeof(T)));
    }
    m_capacity = size;

    // Note: the reported flags of newdata may not match what was requested if
    // it lands on a page already registered with different flags. Freeing first
    // would not guarantee a fresh page either. See flags() docstring for
    // details.

    // Copy over (cap at new size to avoid overflow when downsizing)
    if (m_data != nullptr && m_size > 0)
      std::memcpy(newdata, m_data, std::min(m_size, size) * sizeof(T));

    // Free old
    cudaFreeHost(m_data);

    // Set new
    m_data = newdata;
  }
};

} // namespace containers

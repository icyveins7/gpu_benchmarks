#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

namespace containers {

/**
 * @brief Per-GPU registry that lazily constructs and locks a shared resource
 * of type T for each CUDA device ID.
 *
 * @detail Intended use-case: multiple threads may operate against the same
 * GPU and need exclusive access to some shared per-GPU resource (e.g. a
 * pinned host staging buffer) without every thread allocating its own copy.
 * A thread calls claim(gpuId) to obtain the resource for that GPU; if no
 * resource has been constructed yet for that GPU, one is default-constructed
 * on the spot. The returned Handle holds a lock on that GPU's resource for as
 * long as it is alive, so only one thread may use a given GPU's resource at a
 * time; other threads calling claim() on the same gpuId will block until the
 * Handle is released.
 *
 * T can be anything default-constructible, e.g. thrust::pinned_host_vector,
 * or your own struct wrapping one alongside other per-GPU state you need to
 * share between threads.
 *
 * We are not addressing reentrancy here: claim() is not safe to call again
 * for the same gpuId, on the same thread, while already holding a Handle for
 * that gpuId (e.g. from within a method on T) - std::mutex is not recursive,
 * so this would deadlock.
 *
 * @example Typical usage from downstream code: define your own per-GPU
 * struct, alias a registry for it, then claim/use/release around each
 * critical section.
 * ```
 *  struct MyType {
 *    thrust::pinned_host_vector<float> buffer;
 *    // ... whatever else you need per GPU
 *  };
 *
 *  using CudaRegistry = containers::PerGpuSingleton<MyType>;
 *
 *  void doWork(int gpuId) {
 *    auto handle = CudaRegistry::instance().claim(gpuId);
 *    handle->buffer.resize(1024);
 *    // ... use handle.get() / handle->... / *handle as a MyType&
 *    // resource is unlocked when `handle` goes out of scope
 *  }
 * ```
 * Note that `MyType` must be default-constructible, since it is
 * default-constructed on the first claim() for a given GPU ID; and `handle`
 * must be kept alive (as a named local) for the duration of use, since
 * destroying it releases the lock.
 *
 * @tparam T Resource type stored per GPU. Must be default-constructible.
 */
template <typename T> class PerGpuSingleton {
public:
  /**
   * @brief RAII handle to a locked per-GPU resource, returned by claim().
   * The underlying mutex is released automatically when the handle is
   * destroyed.
   */
  class Handle {
  public:
    Handle(T& resource, std::mutex& mutex)
        : m_resource(resource), m_lock(mutex) {}

    T& get() { return m_resource; }
    T* operator->() { return &m_resource; }
    T& operator*() { return m_resource; }

  private:
    T& m_resource;
    std::unique_lock<std::mutex> m_lock;
  };

  /**
   * @brief Returns the process-wide singleton instance for resource type T.
   * There is one such instance per distinct T used, each maintaining its own
   * map of GPU ID -> resource.
   */
  static PerGpuSingleton<T>& instance() {
    static PerGpuSingleton<T> s_instance;
    return s_instance;
  }

  PerGpuSingleton(const PerGpuSingleton&) = delete;
  PerGpuSingleton& operator=(const PerGpuSingleton&) = delete;

  /**
   * @brief Claims the resource for the given GPU, constructing it if this is
   * the first claim for that GPU ID. Blocks until any other thread currently
   * holding the resource for the same GPU ID releases it.
   *
   * @detail There are two separate mutexes involved, guarding two separate
   * things:
   *  - m_mapMutex guards the map's own structure (find/emplace), since
   *    unordered_map is not safe for concurrent insertion.
   *  - entry.mutex guards usage of the resource itself, so only one thread
   *    can hold a given GPU's resource at a time.
   * getOrCreateEntry() below acquires and releases m_mapMutex on its own,
   * *before* we lock entry.mutex here. That gap is intentional and safe:
   *  - m_map stores unique_ptr<Entry>, not Entry by value, so the Entry
   *    object lives at a fixed heap address for its entire lifetime. Any
   *    rehashing/insertion the map does afterward only moves the
   *    unique_ptr's around, never the Entry it points to, so the reference
   *    returned by getOrCreateEntry() remains valid.
   *  - unique_ptr only controls *ownership* (who frees the Entry), not
   *    mutual exclusion. If two threads call claim() for the same gpuId at
   *    the same time, both may get a reference to the same Entry - that is
   *    fine and expected. Only one of them will actually succeed in locking
   *    entry.mutex (via the Handle constructor below); the other blocks
   *    until the first Handle is destroyed.
   * (This also relies on entries never being erased from m_map; if removal
   * is ever added, this reasoning needs to be revisited.)
   *
   * NOTE: claim() calls cudaSetDevice(gpuId) as its first step, on the
   * calling thread, before doing anything else. This is done here (rather
   * than left to T) since we cannot assume T's constructor/methods will set
   * the device themselves, and since a thread calling claim(gpuId) is
   * expected to operate on that GPU anyway. This means claim() has the side
   * effect of changing the calling thread's current device.
   *
   * @param gpuId CUDA device ID to claim the resource for.
   * @return Handle RAII handle providing access to the resource; releases the
   * lock on destruction.
   */
  Handle claim(int gpuId) {
    cudaError_t err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "PerGpuSingleton::claim: cudaSetDevice failed with error " +
          std::string(cudaGetErrorString(err)));
    }

    Entry& entry = getOrCreateEntry(gpuId);
    return Handle(entry.resource, entry.mutex);
  }

  // TODO: no reset()/resetAll() yet to drop a GPU's entry (or all entries)
  // from the registry. Resources currently live for the lifetime of the
  // singleton (i.e. until process exit); add this if a use-case needs to
  // e.g. tear down and re-initialize a GPU's resource, or reset state
  // between test cases.

private:
  PerGpuSingleton() = default;

  struct Entry {
    T resource;
    std::mutex mutex;
  };

  /**
   * @brief Looks up the entry for gpuId, default-constructing one under the
   * map-wide lock if it does not already exist.
   */
  Entry& getOrCreateEntry(int gpuId) {
    std::lock_guard<std::mutex> mapLock(m_mapMutex);
    auto it = m_map.find(gpuId);
    if (it == m_map.end()) {
      it = m_map.emplace(gpuId, std::make_unique<Entry>()).first;
    }
    return *it->second;
  }

  // Guards m_map itself (insertion of new GPU entries), not the individual
  // resources within it; those are guarded by each Entry's own mutex. See
  // the comment in claim() for why it's safe to release this before locking
  // an individual Entry's mutex.
  std::mutex m_mapMutex;
  std::unordered_map<int, std::unique_ptr<Entry>> m_map;
};

} // namespace containers

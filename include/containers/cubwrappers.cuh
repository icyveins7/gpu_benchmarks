#pragma once

#include <cub/cub.cuh>

#include <cuda/std/functional>

#include <thrust/device_vector.h>

namespace cubw {

template <typename Derived> struct CubWrapper {
public:
  thrust::device_vector<char> d_temp_storage;

  template <typename NumItemsT>
  static void getStorageBytes(NumItemsT num_items) {
    throw std::runtime_error("Not implemented in base class");
  }

  template <typename NumItemsT> void resizeStorage(NumItemsT num_items) {
    size_t temp_storage_bytes =
        static_cast<Derived *>(this)->getStorageBytes(num_items);
    this->d_temp_storage.resize(temp_storage_bytes);
  }

  template <typename... Args> cudaError_t exec(Args &&...args) {
    throw std::runtime_error("Not implemented in base class");
  }

protected:
  CubWrapper() = default;
};

// ===========================
// DeviceRadixSort
// ===========================

namespace DeviceRadixSort {

template <typename KeyT, typename NumItemsT>
struct SortKeys : public CubWrapper<SortKeys<KeyT, NumItemsT>> {
  SortKeys() {}
  SortKeys(NumItemsT num_items) { this->resizeStorage(num_items); }

  static size_t getStorageBytes(NumItemsT num_items) {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                                   (const KeyT *)nullptr, (KeyT *)nullptr,
                                   num_items);
    return temp_storage_bytes;
  }

  cudaError_t exec(const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT num_items,
                   int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceRadixSort::SortKeys(
        this->d_temp_storage.data().get(),
        temp_storage_bytes, // we only do this because it expects a ref
        d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
  }
};

} // namespace DeviceRadixSort

// ===========================
// DeviceMergeSort
// ===========================
namespace DeviceMergeSort {

template <typename KeyInputIteratorT, typename KeyIteratorT, typename OffsetT,
          typename CompareOpT>
struct SortKeysCopy
    : public CubWrapper<
          SortKeysCopy<KeyInputIteratorT, KeyIteratorT, OffsetT, CompareOpT>> {
  SortKeysCopy() {}
  SortKeysCopy(OffsetT num_items) { this->resizeStorage(num_items); }

  static size_t getStorageBytes(OffsetT num_items) {
    size_t temp_storage_bytes = 0;
    CompareOpT compare_op;
    cub::DeviceMergeSort::SortKeysCopy(
        nullptr, temp_storage_bytes, (KeyInputIteratorT) nullptr,
        (KeyIteratorT) nullptr, num_items, compare_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(KeyInputIteratorT d_keys_in, KeyIteratorT d_keys_out,
                   OffsetT num_items, CompareOpT compare_op,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceMergeSort::SortKeysCopy(
        this->d_temp_storage.data().get(), temp_storage_bytes, d_keys_in,
        d_keys_out, num_items, compare_op, stream);
  }
};

} // namespace DeviceMergeSort

} // namespace cubw

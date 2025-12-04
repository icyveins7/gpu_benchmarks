#pragma once

#include <cub/cub.cuh>

#include <cuda/std/functional>

#include <thrust/device_vector.h>

namespace cubw {

/*
Wrapper for CUB; primarily Device-wide primitives
*/

struct CubWrapper {
public:
  thrust::device_vector<char> d_temp_storage;

  virtual size_t getStorageBytes(size_t num_items) = 0;

  template <typename... Args> cudaError_t exec(Args &&...args) {
    throw std::runtime_error("Not implemented in base class");
  }

protected:
  CubWrapper() = default;

  void resizeStorage(size_t num_items) {
    d_temp_storage.resize(getStorageBytes(num_items));
  };
};

// ===========================
// DeviceRadixSort
// ===========================

namespace DeviceRadixSort {

template <typename KeyT, typename NumItemsT>
struct SortKeys : public CubWrapper {
  SortKeys() {}
  SortKeys(NumItemsT num_items) : CubWrapper() {
    resizeStorage((size_t)num_items);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                                   (const KeyT *)nullptr, (KeyT *)nullptr,
                                   (NumItemsT)num_items);
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
struct SortKeysCopy : public CubWrapper {
  SortKeysCopy() {}
  SortKeysCopy(OffsetT num_items) : CubWrapper() {
    resizeStorage((size_t)num_items);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    CompareOpT compare_op;
    cub::DeviceMergeSort::SortKeysCopy(
        nullptr, temp_storage_bytes, (KeyInputIteratorT) nullptr,
        (KeyIteratorT) nullptr, (OffsetT)num_items, compare_op);
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

template <typename KeyInputIteratorT, typename ValueInputIteratorT,
          typename KeyIteratorT, typename ValueIteratorT, typename OffsetT,
          typename CompareOpT>
struct SortPairsCopy : public CubWrapper {
  SortPairsCopy() {}
  SortPairsCopy(OffsetT num_items) : CubWrapper() {
    resizeStorage((size_t)num_items);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    CompareOpT compare_op;
    cub::DeviceMergeSort::SortPairsCopy(
        nullptr, temp_storage_bytes, (KeyInputIteratorT) nullptr,
        (ValueInputIteratorT) nullptr, (KeyIteratorT) nullptr,
        (ValueIteratorT) nullptr, (OffsetT)num_items, compare_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(KeyInputIteratorT d_keys_in, ValueInputIteratorT d_values_in,
                   KeyIteratorT d_keys_out, ValueIteratorT d_values_out,
                   OffsetT num_items, CompareOpT compare_op,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceMergeSort::SortPairsCopy(
        this->d_temp_storage.data().get(), temp_storage_bytes, d_keys_in,
        d_values_in, d_keys_out, d_values_out, num_items, compare_op, stream);
  }
};

} // namespace DeviceMergeSort

} // namespace cubw

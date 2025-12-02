#pragma once

#include <cub/cub.cuh>

#include <thrust/device_vector.h>

namespace cubw {

struct CubWrapper {
  thrust::device_vector<char> d_temp_storage;

  template <typename NumItemsT>
  static void getStorageBytes(NumItemsT num_items) {
    throw std::runtime_error("Not implemented in base class");
  }

  template <typename NumItemsT> void resizeStorage(NumItemsT num_items) {
    throw std::runtime_error("Not implemented in base class");
  }

  template <typename... Args> void exec(Args &&...args) {
    throw std::runtime_error("Not implemented in base class");
  }
};

template <typename KeyT, typename NumItemsT>
struct SortKeys : public CubWrapper {
  SortKeys() {}
  SortKeys(NumItemsT num_items) { getStorageBytes(num_items); }

  static size_t getStorageBytes(NumItemsT num_items) {
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                                   (const KeyT *)nullptr, (KeyT *)nullptr,
                                   num_items);
    return temp_storage_bytes;
  }

  void resizeStorage(NumItemsT num_items) {
    size_t temp_storage_bytes = getStorageBytes(num_items);
    this->d_temp_storage.resize(temp_storage_bytes);
  }

  void exec(const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT num_items,
            int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
            cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage.data().get(),
        temp_storage_bytes, // we only do this because it expects a ref
        d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
  }
};

} // namespace cubw

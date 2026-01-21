#pragma once

#include <cub/cub.cuh>

#include <cuda/std/functional>

#include <thrust/device_vector.h>

#include <containers/streams.cuh>

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
__global__ void SortKeysCDP_kernel(void *d_temp_storage,
                                   size_t temp_storage_bytes,
                                   const KeyT *d_keys_in, KeyT *d_keys_out,
                                   NumItemsT *d_num_items, int begin_bit = 0,
                                   int end_bit = sizeof(KeyT) * 8) {
  if (blockIdx.x != 0)
    return;

  NumItemsT num_items = *d_num_items;

  // Launch CUB routine
  if (threadIdx.x == 0) {
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   d_keys_in, d_keys_out, num_items, begin_bit,
                                   end_bit);
  }
}

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
                   cudaStream_t stream = 0, int begin_bit = 0,
                   int end_bit = sizeof(KeyT) * 8) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceRadixSort::SortKeys(
        this->d_temp_storage.data().get(),
        temp_storage_bytes, // we only do this because it expects a ref
        d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
  }

  void cdp_exec(const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT *d_num_items,
                cudaStream_t stream = 0, int begin_bit = 0,
                int end_bit = sizeof(KeyT) * 8) {

    size_t temp_storage_bytes = this->d_temp_storage.size();
    SortKeysCDP_kernel<KeyT, NumItemsT><<<1, 1, 0, stream>>>(
        this->d_temp_storage.data().get(), temp_storage_bytes, d_keys_in,
        d_keys_out, d_num_items, begin_bit, end_bit);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
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
    // Default-construct all types; pointers automatically become nullptrs
    CompareOpT compare_op;
    KeyInputIteratorT keyinput{};
    KeyIteratorT keyoutput{};
    cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_storage_bytes, keyinput,
                                       keyoutput, (OffsetT)num_items,
                                       compare_op);
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
    // Default-construct all types; pointers automatically become nullptrs
    CompareOpT compare_op;
    KeyInputIteratorT keyinput{};
    KeyIteratorT keyoutput{};
    ValueInputIteratorT valueinput{};
    ValueIteratorT valueoutput{};
    cub::DeviceMergeSort::SortPairsCopy(nullptr, temp_storage_bytes, keyinput,
                                        valueinput, keyoutput, valueoutput,
                                        (OffsetT)num_items, compare_op);
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

// ===========================
// DeviceSelect
// ===========================
namespace DeviceSelect {

template <typename IteratorT, typename NumSelectedIteratorT, typename SelectOp>
struct IfInPlace : public CubWrapper {
  IfInPlace() {}
  IfInPlace(int num_items) : CubWrapper() { resizeStorage((size_t)num_items); }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    SelectOp select_op{};
    IteratorT data{};
    NumSelectedIteratorT num_selected{};
    cub::DeviceSelect::If(nullptr, temp_storage_bytes, data, num_selected,
                          num_items, select_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(IteratorT d_in, NumSelectedIteratorT d_num_selected,
                   ::cuda::std::int64_t num_items, SelectOp select_op,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceSelect::If(this->d_temp_storage.data().get(),
                                 temp_storage_bytes, d_in, d_num_selected,
                                 num_items, select_op, stream);
  }
};

template <typename InputIteratorT, typename OutputIteratorT,
          typename NumSelectedIteratorT, typename SelectOp>
struct If : public CubWrapper {
  If() {}
  If(int num_items) : CubWrapper() { resizeStorage((size_t)num_items); }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    SelectOp select_op{};
    InputIteratorT input{};
    OutputIteratorT output{};
    NumSelectedIteratorT num_selected{};
    cub::DeviceSelect::If(nullptr, temp_storage_bytes, input, output,
                          num_selected, num_items, select_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(InputIteratorT d_in, OutputIteratorT d_out,
                   NumSelectedIteratorT d_num_selected,
                   ::cuda::std::int64_t num_items, SelectOp select_op,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceSelect::If(this->d_temp_storage.data().get(),
                                 temp_storage_bytes, d_in, d_out,
                                 d_num_selected, num_items, select_op, stream);
  }
};

} // namespace DeviceSelect

namespace DeviceScan {

template <typename KeysInputIteratorT, typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename EqualityOpT = ::cuda::std::equal_to<>,
          typename NumItemsT = uint32_t>
struct InclusiveSumByKey : public CubWrapper {
  InclusiveSumByKey() {}
  InclusiveSumByKey(NumItemsT num_items) : CubWrapper() {
    resizeStorage((size_t)num_items);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    EqualityOpT equality_op{};
    KeysInputIteratorT inputkeys{};
    ValuesInputIteratorT input{};
    ValuesOutputIteratorT output{};
    cub::DeviceScan::InclusiveSumByKey(nullptr, temp_storage_bytes, inputkeys,
                                       input, output, num_items, equality_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(KeysInputIteratorT d_keys_in,
                   ValuesInputIteratorT d_values_in,
                   ValuesOutputIteratorT d_values_out, NumItemsT num_items,
                   EqualityOpT equality_op = EqualityOpT(),
                   cudaStream_t stream = 0) {

    size_t temp_storage_bytes = this->d_temp_storage.size();

    return cub::DeviceScan::InclusiveSumByKey(
        this->d_temp_storage.data().get(), temp_storage_bytes, d_keys_in,
        d_values_in, d_values_out, num_items, equality_op, stream);
  }
};

} // namespace DeviceScan

} // namespace cubw

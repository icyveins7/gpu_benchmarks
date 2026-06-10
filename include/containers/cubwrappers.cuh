#pragma once

#include <cub/cub.cuh>

#include <cuda/std/functional>

#include <thrust/device_vector.h>

#include <type_traits>

#include <containers/stream_ordered_storage.cuh>
#include <containers/streams.cuh>

namespace cubw {

/*
Wrapper for CUB; primarily Device-wide primitives
*/

template <bool StreamOrdered = false>
struct CubWrapper {
public:
  using storage_t = std::conditional_t<StreamOrdered,
      containers::StreamOrderedDeviceStorage<char>,
      thrust::device_vector<char>>;

  storage_t d_temp_storage;

  virtual size_t getStorageBytes(size_t num_items) = 0;

  template <typename... Args> cudaError_t exec(Args &&...args) {
    throw std::runtime_error("Not implemented in base class");
  }

protected:
  CubWrapper() = default;

  void resizeStorage(size_t num_items, cudaStream_t stream = 0) {
    size_t bytes = getStorageBytes(num_items);
    if constexpr (StreamOrdered) {
      if (d_temp_storage.data() == nullptr)
        d_temp_storage.initialize(bytes, stream);
      else
        d_temp_storage.resizeWithoutCopy(bytes);
    } else {
      d_temp_storage.resize(bytes);
    }
  }

  void *storagePtr() {
    if constexpr (StreamOrdered)
      return d_temp_storage.data();
    else
      return d_temp_storage.data().get();
  }
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

template <typename KeyT, typename NumItemsT, bool StreamOrdered = false>
struct SortKeys : public CubWrapper<StreamOrdered> {
  SortKeys() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  SortKeys(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  SortKeys(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
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
        this->storagePtr(),
        temp_storage_bytes, // we only do this because it expects a ref
        d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
  }

  void cdp_exec(const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT *d_num_items,
                cudaStream_t stream = 0, int begin_bit = 0,
                int end_bit = sizeof(KeyT) * 8) {

    size_t temp_storage_bytes = this->d_temp_storage.size();
    SortKeysCDP_kernel<KeyT, NumItemsT><<<1, 1, 0, stream>>>(
        this->storagePtr(), temp_storage_bytes, d_keys_in,
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
          typename CompareOpT, bool StreamOrdered = false>
struct SortKeysCopy : public CubWrapper<StreamOrdered> {
  SortKeysCopy() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  SortKeysCopy(OffsetT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  SortKeysCopy(OffsetT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
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
        this->storagePtr(), temp_storage_bytes, d_keys_in,
        d_keys_out, num_items, compare_op, stream);
  }
};

template <typename KeyInputIteratorT, typename ValueInputIteratorT,
          typename KeyIteratorT, typename ValueIteratorT, typename OffsetT,
          typename CompareOpT, bool StreamOrdered = false>
struct SortPairsCopy : public CubWrapper<StreamOrdered> {
  SortPairsCopy() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  SortPairsCopy(OffsetT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  SortPairsCopy(OffsetT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
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
        this->storagePtr(), temp_storage_bytes, d_keys_in,
        d_values_in, d_keys_out, d_values_out, num_items, compare_op, stream);
  }
};

} // namespace DeviceMergeSort

// ===========================
// DeviceSelect
// ===========================
namespace DeviceSelect {

template <typename IteratorT, typename NumSelectedIteratorT, typename SelectOp,
          bool StreamOrdered = false>
struct IfInPlace : public CubWrapper<StreamOrdered> {
  IfInPlace() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  IfInPlace(int num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  IfInPlace(int num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

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
    return cub::DeviceSelect::If(this->storagePtr(),
                                 temp_storage_bytes, d_in, d_num_selected,
                                 num_items, select_op, stream);
  }
};

template <typename InputIteratorT, typename OutputIteratorT,
          typename NumSelectedIteratorT, typename SelectOp,
          bool StreamOrdered = false>
struct If : public CubWrapper<StreamOrdered> {
  If() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  If(int num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  If(int num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

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
    return cub::DeviceSelect::If(this->storagePtr(),
                                 temp_storage_bytes, d_in, d_out,
                                 d_num_selected, num_items, select_op, stream);
  }
};

} // namespace DeviceSelect

// ===========================
// DeviceScan
// ===========================
namespace DeviceScan {

template <typename KeysInputIteratorT, typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename EqualityOpT = ::cuda::std::equal_to<>,
          typename NumItemsT = uint32_t, bool StreamOrdered = false>
struct InclusiveSumByKey : public CubWrapper<StreamOrdered> {
  InclusiveSumByKey() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  InclusiveSumByKey(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  InclusiveSumByKey(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
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
        this->storagePtr(), temp_storage_bytes, d_keys_in,
        d_values_in, d_values_out, num_items, equality_op, stream);
  }
};

template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT,
          bool StreamOrdered = false>
struct ExclusiveSum : public CubWrapper<StreamOrdered> {
  ExclusiveSum() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  ExclusiveSum(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  ExclusiveSum(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    InputIteratorT input{};
    OutputIteratorT output{};
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, input, output,
                                  num_items);
    return temp_storage_bytes;
  }

  cudaError_t exec(InputIteratorT d_in, OutputIteratorT d_out,
                   NumItemsT num_items, cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceScan::ExclusiveSum(this->storagePtr(),
                                         temp_storage_bytes, d_in, d_out,
                                         num_items, stream);
  }
};

template <typename IteratorT, typename NumItemsT, bool StreamOrdered = false>
struct ExclusiveSumInPlace : public CubWrapper<StreamOrdered> {
  ExclusiveSumInPlace() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  ExclusiveSumInPlace(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  ExclusiveSumInPlace(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    IteratorT inout{};
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, inout,
                                  num_items);
    return temp_storage_bytes;
  }

  cudaError_t exec(IteratorT d_inout, NumItemsT num_items,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceScan::ExclusiveSum(this->storagePtr(),
                                         temp_storage_bytes, d_inout, num_items,
                                         stream);
  }
};

} // namespace DeviceScan

// ===========================
// DeviceSegmentedReduce
// ===========================
namespace DeviceSegmentedReduce {

template <typename InputIteratorT, typename OutputIteratorT,
          typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
          typename ReductionOpT, typename T, bool StreamOrdered = false>
struct Reduce : public CubWrapper<StreamOrdered> {
  Reduce() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  Reduce(int num_segments) {
    this->resizeStorage((size_t)num_segments);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  Reduce(int num_segments, cudaStream_t stream) {
    this->resizeStorage((size_t)num_segments, stream);
  }

  size_t getStorageBytes(size_t num_segments) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    ReductionOpT reduction_op{};
    InputIteratorT input{};
    OutputIteratorT output{};
    BeginOffsetIteratorT begin_offsets{};
    EndOffsetIteratorT end_offsets{};
    T initialValue{};
    cub::DeviceSegmentedReduce::Reduce(nullptr, temp_storage_bytes, input,
                                       output, num_segments, begin_offsets,
                                       end_offsets, reduction_op, initialValue);
    return temp_storage_bytes;
  }

  cudaError_t exec(InputIteratorT d_in, OutputIteratorT d_out,
                   int64_t num_segments, BeginOffsetIteratorT d_begin_offsets,
                   EndOffsetIteratorT d_end_offsets, ReductionOpT reduction_op,
                   T initialValue, cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceSegmentedReduce::Reduce(
        this->storagePtr(), temp_storage_bytes, d_in, d_out,
        num_segments, d_begin_offsets, d_end_offsets, reduction_op,
        initialValue, stream);
  }
};

} // namespace DeviceSegmentedReduce

// ===========================
// DeviceRunLengthEncode
// ===========================
namespace DeviceRunLengthEncode {

template <typename InputIteratorT, typename UniqueOutputIteratorT,
          typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT,
          typename NumItemsT, bool StreamOrdered = false>
struct Encode : public CubWrapper<StreamOrdered> {
  Encode() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  Encode(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  Encode(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    // Default-construct all types; pointers automatically become nullptrs
    InputIteratorT input{};
    UniqueOutputIteratorT unique_output{};
    LengthsOutputIteratorT lengths_output{};
    NumRunsOutputIteratorT num_runs_output{};
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes, input,
                                       unique_output, lengths_output,
                                       num_runs_output, num_items);
    return temp_storage_bytes;
  }

  cudaError_t exec(InputIteratorT d_in, UniqueOutputIteratorT d_unique_out,
                   LengthsOutputIteratorT d_lengths_out,
                   NumRunsOutputIteratorT d_num_runs_out, NumItemsT num_items,
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceRunLengthEncode::Encode(
        this->storagePtr(), temp_storage_bytes, d_in,
        d_unique_out, d_lengths_out, d_num_runs_out, num_items, stream);
  }
};

} // namespace DeviceRunLengthEncode

} // namespace cubw

#pragma once

#include <cub/cub.cuh>

#include <cuda/std/functional>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <type_traits>

#include <containers/stream_ordered_storage.cuh>
#include <containers/streams.cuh>

namespace cubw {

/*
Wrapper for CUB; primarily Device-wide primitives
*/

template <bool StreamOrdered = false> struct CubWrapper {
public:
  using storage_t =
      std::conditional_t<StreamOrdered,
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
    SortKeysCDP_kernel<KeyT, NumItemsT>
        <<<1, 1, 0, stream>>>(this->storagePtr(), temp_storage_bytes, d_keys_in,
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
        this->storagePtr(), temp_storage_bytes, d_keys_in, d_keys_out,
        num_items, compare_op, stream);
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
        this->storagePtr(), temp_storage_bytes, d_keys_in, d_values_in,
        d_keys_out, d_values_out, num_items, compare_op, stream);
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
    return cub::DeviceSelect::If(this->storagePtr(), temp_storage_bytes, d_in,
                                 d_num_selected, num_items, select_op, stream);
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
    return cub::DeviceSelect::If(this->storagePtr(), temp_storage_bytes, d_in,
                                 d_out, d_num_selected, num_items, select_op,
                                 stream);
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
        this->storagePtr(), temp_storage_bytes, d_keys_in, d_values_in,
        d_values_out, num_items, equality_op, stream);
  }
};

template <typename KeysInputIteratorT, typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT, typename ScanOpT,
          typename EqualityOpT = ::cuda::std::equal_to<>,
          typename NumItemsT = uint32_t, bool StreamOrdered = false>
struct InclusiveScanByKey : public CubWrapper<StreamOrdered> {
  InclusiveScanByKey() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  InclusiveScanByKey(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  InclusiveScanByKey(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    ScanOpT scan_op{};
    EqualityOpT equality_op{};
    KeysInputIteratorT inputkeys{};
    ValuesInputIteratorT input{};
    ValuesOutputIteratorT output{};
    cub::DeviceScan::InclusiveScanByKey(nullptr, temp_storage_bytes, inputkeys,
                                        input, output, scan_op, num_items,
                                        equality_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(KeysInputIteratorT d_keys_in,
                   ValuesInputIteratorT d_values_in,
                   ValuesOutputIteratorT d_values_out, ScanOpT scan_op,
                   NumItemsT num_items, EqualityOpT equality_op = EqualityOpT(),
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceScan::InclusiveScanByKey(
        this->storagePtr(), temp_storage_bytes, d_keys_in, d_values_in,
        d_values_out, scan_op, num_items, equality_op, stream);
  }
};

template <typename KeysInputIteratorT, typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT, typename ScanOpT, typename InitValueT,
          typename EqualityOpT = ::cuda::std::equal_to<>,
          typename NumItemsT = uint32_t, bool StreamOrdered = false>
struct ExclusiveScanByKey : public CubWrapper<StreamOrdered> {
  ExclusiveScanByKey() {}

  template <bool S = StreamOrdered, std::enable_if_t<!S, int> = 0>
  ExclusiveScanByKey(NumItemsT num_items) {
    this->resizeStorage((size_t)num_items);
  }

  template <bool S = StreamOrdered, std::enable_if_t<S, int> = 0>
  ExclusiveScanByKey(NumItemsT num_items, cudaStream_t stream) {
    this->resizeStorage((size_t)num_items, stream);
  }

  size_t getStorageBytes(size_t num_items) override {
    size_t temp_storage_bytes = 0;
    ScanOpT scan_op{};
    InitValueT init_value{};
    EqualityOpT equality_op{};
    KeysInputIteratorT inputkeys{};
    ValuesInputIteratorT input{};
    ValuesOutputIteratorT output{};
    cub::DeviceScan::ExclusiveScanByKey(nullptr, temp_storage_bytes, inputkeys,
                                        input, output, scan_op, init_value,
                                        num_items, equality_op);
    return temp_storage_bytes;
  }

  cudaError_t exec(KeysInputIteratorT d_keys_in,
                   ValuesInputIteratorT d_values_in,
                   ValuesOutputIteratorT d_values_out, ScanOpT scan_op,
                   InitValueT init_value, NumItemsT num_items,
                   EqualityOpT equality_op = EqualityOpT(),
                   cudaStream_t stream = 0) {
    size_t temp_storage_bytes = this->d_temp_storage.size();
    return cub::DeviceScan::ExclusiveScanByKey(
        this->storagePtr(), temp_storage_bytes, d_keys_in, d_values_in,
        d_values_out, scan_op, init_value, num_items, equality_op, stream);
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
    return cub::DeviceScan::ExclusiveSum(this->storagePtr(), temp_storage_bytes,
                                         d_in, d_out, num_items, stream);
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
    return cub::DeviceScan::ExclusiveSum(this->storagePtr(), temp_storage_bytes,
                                         d_inout, num_items, stream);
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
        this->storagePtr(), temp_storage_bytes, d_in, d_out, num_segments,
        d_begin_offsets, d_end_offsets, reduction_op, initialValue, stream);
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
        this->storagePtr(), temp_storage_bytes, d_in, d_unique_out,
        d_lengths_out, d_num_runs_out, num_items, stream);
  }
};

} // namespace DeviceRunLengthEncode

// ===========================
// Helpers
// ===========================
namespace helpers {

/**
 * @brief Transform functor that maps a flat element index to its row number.
 * Produces segment keys for use with segmented CUB/Thrust calls like
 * InclusiveScanByKey, where each row is an independent segment.
 *
 * @example
 * For a 2x2 image (width=2), a counting_iterator
 * 0 1
 * 2 3
 * becomes row keys
 * 0 0
 * 1 1
 *
 * @tparam Trow   Integer type; choose type that can hold max rows/columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 */
template <typename Trow, typename Tcount = int32_t> struct IndexToRowFunctor {
  static_assert(std::is_integral_v<Trow>, "Trow must be an integer type");
  static_assert(std::is_integral_v<Tcount>, "Tcount must be an integer type");
  Trow width = 0; // width of image

  __host__ __device__ __forceinline__ Trow operator()(Tcount index) const {
    return index / width;
  }
};

/**
 * @brief Returns a transform_iterator that maps flat element indices to row
 * keys, suitable as the key iterator for segmented CUB/Thrust calls.
 *
 * @detail The result of this function is an iterator that can be passed
 * directly to the key input iterator of segmented CUB calls like
 * InclusiveScanByKey.
 *
 * @tparam Trow   Integer type; choose type that can hold max rows/columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 * @param width Width of the image
 */
template <typename Trow, typename Tcount = int32_t>
auto makeRowIndexIterator(Trow width) {
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator<Tcount>(0),
      IndexToRowFunctor<Trow, Tcount>{width});
}

/**
 * @brief Transform functor that maps a flat element index to its column number.
 * Produces column keys for use with segmented CUB/Thrust calls like
 * InclusiveScanByKey, where each column is an independent segment.
 *
 * @example
 * For a 2x2 image (width=2), a counting_iterator
 * 0 1
 * 2 3
 * becomes col keys
 * 0 1
 * 0 1
 *
 * @tparam Tcol Integer index type; choose type that can hold max columns
 * @tparam Tcount Integer index type; choose type that can hold max elements
 */
template <typename Tcol, typename Tcount = int32_t> struct IndexToColFunctor {
  static_assert(std::is_integral_v<Tcol>, "T must be an integer type");
  static_assert(std::is_integral_v<Tcount>, "Tcount must be an integer type");
  Tcol width = 0; // width of image

  __host__ __device__ __forceinline__ Tcol operator()(Tcount index) const {
    return index % width;
  }
};

/**
 * @brief Returns a transform_iterator that maps flat element indices to column
 * keys, suitable as the key iterator for segmented CUB/Thrust calls.
 *
 * @tparam Tcol   Integer type; choose type that can hold max columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 * @param width Width of the image
 */
template <typename Tcol, typename Tcount = int32_t>
auto makeColIndexIterator(Tcol width) {
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator<Tcount>(0),
      IndexToColFunctor<Tcol, Tcount>{width});
}

/**
 * @brief Transform functor that maps a flat index over a subset of selected
 * rows to a flat memory index into the full image. Selects every row_stride-th
 * row (rows 0, row_stride, 2*row_stride, ...). Suitable for use with
 * thrust::make_permutation_iterator to read/write those rows.
 *
 * @example
 * For width=10, row_stride=4 (accessing rows 0, 4, 8, ...):
 * selected row 0 (indices  0- 9) → flat offsets  0- 9
 * selected row 1 (indices 10-19) → flat offsets 40-49
 * selected row 2 (indices 20-29) → flat offsets 80-89
 *
 * @tparam Trow   Integer type; choose type that can hold max rows/columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 */
template <typename Trow, typename Tcount = int32_t>
struct IndexToRowStridedIndexFunctor {
  static_assert(std::is_integral_v<Trow>, "T must be an integer type");
  static_assert(std::is_integral_v<Tcount>, "Tcount must be an integer type");
  Trow width = 0; // width of image
  Trow row_stride =
      0; // every row_stride-th row is selected (e.g. 4 → rows 0,4,8,...)

  __host__ __device__ __forceinline__ Tcount operator()(Tcount index) const {
    return (index / width) * row_stride * width + (index % width);
  }
};

/**
 * @brief Returns a transform_iterator that maps flat indices over selected rows
 * to flat memory indices into the full image, suitable as the index argument to
 * thrust::make_permutation_iterator.
 *
 * @tparam Trow   Integer type; choose type that can hold max rows/columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 * @param width     Width of the image
 * @param row_stride Every row_stride-th row is selected (e.g. 4 → rows
 * 0,4,8,...)
 */
template <typename Trow, typename Tcount = int32_t>
auto makeRowStridedIndexIterator(Trow width, Trow row_stride) {
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator<Tcount>(0),
      IndexToRowStridedIndexFunctor<Trow, Tcount>{width, row_stride});
}

/**
 * @brief Returns a permutation_iterator over @p ptr that accesses every
 * row_stride-th row of the image. Convenience wrapper around
 * makeRowStridedIndexIterator.
 *
 * @example
 * For a 4x3 image (M=4 rows, N=3 cols) with row_stride=2,
 * makeRowStridedIterator(ptr, 3, 2) accesses rows 0 and 2:
 *
 *   row 0: [  0,  1,  2 ]  ← iter[0], iter[1], iter[2]
 *   row 1: [  3,  4,  5 ]
 *   row 2: [  6,  7,  8 ]  ← iter[3], iter[4], iter[5]
 *   row 3: [  9, 10, 11 ]
 *
 * For InclusiveScanByKey over those 2 selected rows (6 elements total),
 * this is the value iterator. The matching key iterator, which resets the
 * scan at each row boundary, is makeRowIndexIterator:
 *
 *   auto vals = makeRowStridedIterator(ptr, 3, 2);
 *   auto keys = makeRowIndexIterator<int>(3);   // keys: 0 0 0 1 1 1
 *   cubw::DeviceScan::InclusiveScanByKey<...> scan(6);
 *   scan.exec(keys, vals, out, 6);
 *
 * @tparam Ptr    Pointer type to the image data
 * @tparam Trow   Integer type; choose type that can hold max rows/columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 * @param ptr        Pointer to the start of the image
 * @param width      Width of the image (N)
 * @param row_stride Every row_stride-th row is selected (e.g. 4 → rows
 * 0,4,8,...)
 */
template <typename Ptr, typename Trow, typename Tcount = int32_t>
auto makeRowStridedIterator(Ptr ptr, Trow width, Trow row_stride) {
  return thrust::make_permutation_iterator(
      ptr, makeRowStridedIndexIterator<Trow, Tcount>(width, row_stride));
}

/**
 * @brief Common transform functor to turn a counting iterator into a reversed
 * row index based on a width. Useful for segmented calls like
 * InclusiveScanByKey, but backwards (speed is equivalent to forwards scan since
 * still coalesced). Intended to be used with a thrust::make_transform_iterator.
 *
 * @example
 * For a 2x2 image, a counting_iterator
 * 3 2
 * 1 0
 * becomes
 * 0 0
 * 1 1
 *
 * @tparam Trow   Integer type; choose type that can hold max rows/columns
 * @tparam Tcount Integer type; choose type that can hold max elements
 */
template <typename Trow, typename Tcount = int32_t>
struct ReverseIndexToRowFunctor {
  Trow width;
  Tcount total; // height * width of image

  __host__ __device__ __forceinline__ Trow operator()(Tcount i) const {
    return (total - 1 - i) / width;
  }
};

} // namespace helpers

} // namespace cubw

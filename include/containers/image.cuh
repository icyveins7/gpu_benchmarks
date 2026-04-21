#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace containers {

/**
 * @brief Container to hold an image pointer to data and its width/height,
 * along with helpful methods for accessing the data.
 * Primarily used with kernels, which cannot accept RAII structs like
 * device_vectors. See DeviceImageStorage for associated RAII struct.
 *
 * @tparam Tdata Type of data
 * @tparam Tidx Type of indices used in methods
 * @param data Pointer to data, usually a device pointer
 * @param width Width of image in pixels
 * @param height Height of image in pixels
 * @param bytesPerRow Offset of each row in bytes
 */
template <typename Tdata, typename Tidx = int> struct Image {
  static_assert(std::is_integral_v<Tidx>, "Tidx must be an integer type");

  Tdata *data;
  Tidx width;
  Tidx height;
  Tidx bytesPerRow;

  /**
   * @brief Constructor. Automatically sets bytePerRow, assuming no padding.
   * Otherwise, list-initialization for the members can/should be used directly.
   */
  __host__ __device__ Image(Tdata *_data, const Tidx _width, const Tidx _height)
      : data(_data), width(_width), height(_height),
        bytesPerRow(width * sizeof(Tdata)) {}

  /**
   * @brief Does the same thing as the constructor. Useful for delayed
   * initialization, like in conditional blocks.
   */
  __host__ __device__ void initialize(Tdata *_data, const Tidx _width,
                                      const Tidx _height) {
    data = _data;
    width = _width;
    height = _height;
    bytesPerRow = width * sizeof(Tdata);
  }

  __host__ __device__ bool rowIsValid(Tidx y) const {
    return y >= 0 && y < height;
  }
  __host__ __device__ bool colIsValid(Tidx x) const {
    return x >= 0 && x < width;
  }

  /**
   * @brief Returns a pointer to the specified row.
   */
  __host__ __device__ Tdata *row(Tidx y) const {
    return (Tdata *)((uint8_t *)(data) + y * bytesPerRow);
  }

  /**
   * @brief Returns a reference to the specified element.
   */
  __host__ __device__ Tdata &at(Tidx y, Tidx x) { return row(y)[x]; }

  /**
   * @brief Returns a const reference to the specified element.
   */
  __host__ __device__ const Tdata &at(Tidx y, Tidx x) const {
    return row(y)[x];
  }

  /**
   * @brief Returns a reference to the last element of a row.

   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ Tdata &rowEnd(Tidx y) { return row(y)[width - 1]; }

  /**
   * @brief Returns a const reference to the last element of a row.

   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ const Tdata &rowEnd(Tidx y) const {
    return row(y)[width - 1];
  }

  /**
   * @brief Returns a reference to the last element of a column.
   *
   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ Tdata &colEnd(Tidx x) { return row(height - 1)[x]; }

  /**
   * @brief Returns a const reference to the last element of a column.
   *
   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ const Tdata &colEnd(Tidx x) const {
    return row(height - 1)[x];
  }

  /**
   * @brief Returns the element by value if in range, otherwise returns the
   * specified default value.
   */
  __host__ __device__ Tdata atWithDefault(Tidx y, Tidx x,
                                          Tdata defaultValue = 0) const {
    return rowIsValid(y) && colIsValid(x) ? row(y)[x] : defaultValue;
  }

  /**
   * @brief Returns the element by value if in range, otherwise returns the
   * specified pointer's dereferenced value. This prevents an early memory
   * access if it is not required e.g. use row end values for indices extending
   * past the width.

   * @param defaultValuePtr Pointer location to dereference in the case of out
   of bounds indices. Defaults to nullptr, which will instead access the first
   value from *data.
   */
  __host__ __device__ Tdata atWithDefaultPointer(
      Tidx y, Tidx x, const Tdata *defaultValuePtr = nullptr) const {
    // Prevent nullptr access by defaulting to start of data
    const Tdata *ptr = defaultValuePtr == nullptr ? data : defaultValuePtr;
    return rowIsValid(y) && colIsValid(x) ? row(y)[x] : *ptr;
  }
};

/**
 * @brief A data structure to help in referencing rectangular tile sections from
 * original Image data.
 *
 * @detail The targeted use-case is to allow indexing inside kernels into a
 * shared memory tile, while still using indices from the original (larger)
 * image.
 *
 * @example Original image is 4x4, and we create a tile using the centre 2x2.
 * X X X X
 * X T T X ---> A B
 * X T T X ---> C D
 * X X X X
 *
 * tile.at(1,1) -> A
 * tile.at(1,2) -> B
 * tile.at(2,1) -> C
 * tile.at(2,2) -> D
 *
 * Any other element should be out of bounds, and should be pre-checked just as
 * in the Image struct, e.g.
 *
 * tile.rowIsValid(0) -> false
 * tile.colIsValid(3) -> false
 *
 * The beauty of this is that once we have established the bounds of the tile,
 * we should be able to copy the tile from a global Image to a shared memory
 * ImageTile via a simple loop over:
 *
 * tile.at(i, j) = image.at(i, j);
 * Note how there are no offsets in any arguments.
 *
 * @tparam Tdata Type of data
 * @tparam Tidx Type of indices used in methods
 * @param data Pointer to data, usually a device pointer
 * @param width Width of tile in pixels
 * @param height Height of tile in pixels
 * @param bytesPerRow Offset of each row of the tile in bytes
 * @param startRow Starting row of the tile
 * @param startCol Starting col of the tile
 * @return
 */
template <typename Tdata, typename Tidx = int>
struct ImageTile : Image<Tdata, Tidx> {
  // Additional trackers
  Tidx startRow;
  Tidx startCol;

  __host__ __device__ ImageTile(Tdata *_data, const Tidx _width,
                                const Tidx _height, const Tidx _startRow,
                                const Tidx _startCol)
      : Image<Tdata, Tidx>(_data, _width, _height), startRow(_startRow),
        startCol(_startCol) {}

  __host__ __device__ void initialize(Tdata *_data, const Tidx _width,
                                      const Tidx _height, const Tidx _startRow,
                                      const Tidx _startCol) {
    Image<Tdata>::initialize(_data, _width, _height);
    this->startRow = _startRow;
    this->startCol = _startCol;
  }

  /**
   * @brief Convenience method to fill a tile from an Image struct within a
   * thread block.
   * This method does not syncthreads() for you!
   *
   * @param img Input Image struct
   */
  __device__ void fillFromImage(const Image<const Tdata> &img,
                                const Tdata init = 0) {
    for (int ty = threadIdx.y; ty < this->height; ty += blockDim.y) {
      int y = ty + startRow;
      bool yValid = rowIsValid(y) && img.rowIsValid(y);
      for (int tx = threadIdx.x; tx < this->width; tx += blockDim.x) {
        int x = tx + startCol;
        bool xValid = colIsValid(x) && img.colIsValid(x);
        Tdata value = init;
        if (yValid && xValid) {
          value = img.at(y, x);
        }
        this->at(y, x) = value;
      }
    }
  }

  __host__ __device__ bool rowIsValid(Tidx y) const {
    return y >= startRow && y < this->height + startRow;
  }
  __host__ __device__ bool colIsValid(Tidx x) const {
    return x >= startCol && x < this->width + startCol;
  }

  /**
   * @brief Converts the global image row index to the local tile row index.
   * You should not need to call this manually.
   */
  __host__ __device__ Tidx internalRow(Tidx y) const { return y - startRow; }
  /**
   * @brief Converts the global image col index to the local tile col index.
   * You should not need to call this manually.
   */
  __host__ __device__ Tidx internalCol(Tidx x) const { return x - startCol; }

  /**
   * @brief Returns a reference to the specified element.
   *
   * @param y Row index with respect to the original (larger) image
   * @param x Col index with respect to the original (larger) image
   * @return A const reference to the specified element
   */
  __host__ __device__ Tdata &at(Tidx y, Tidx x) {
    return this->row(internalRow(y))[internalCol(x)];
  }

  /**
   * @brief Returns a const reference to the specified element.
   *
   * @param y Row index with respect to the original (larger) image
   * @param x Col index with respect to the original (larger) image
   * @return A const reference to the specified element
   */
  __host__ __device__ const Tdata &at(Tidx y, Tidx x) const {
    return this->row(internalRow(y))[internalCol(x)];
  }

  /**
   * @brief Returns a reference to the last element of a row (in the tile).

   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ Tdata &rowEnd(Tidx y) {
    return this->row(internalRow(y))[this->width - 1];
  }

  /**
   * @brief Returns a const reference to the last element of a row (in the
   tile).

   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ const Tdata &rowEnd(Tidx y) const {
    return this->row(internalRow(y))[this->width - 1];
  }

  /**
   * @brief Returns a reference to the last element of a column (in the tile).
   *
   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ Tdata &colEnd(Tidx x) {
    return this->row(this->height - 1)[internalCol(x)];
  }

  /**
   * @brief Returns a const reference to the last element of a column (in the
   * tile).
   *
   * @detail Useful for scenarios where out of bounds access should be treated
   * as using the last element i.e. 'same-element extrapolation'.
   */
  __host__ __device__ const Tdata &colEnd(Tidx x) const {
    return this->row(this->height - 1)[internalCol(x)];
  }

  /**
   * @brief Returns the element by value if in range, otherwise returns the
   * specified default value.
   */
  __host__ __device__ Tdata atWithDefault(Tidx y, Tidx x,
                                          Tdata defaultValue = 0) const {
    return rowIsValid(y) && colIsValid(x)
               ? this->row(internalRow(y))[internalCol(x)]
               : defaultValue;
  }

  /**
   * @brief Returns the element by value if in range, otherwise returns the
   * specified pointer's dereferenced value. This prevents an early memory
   * access if it is not required e.g. use row end values for indices extending
   * past the width.

   * @param defaultValuePtr Pointer location to dereference in the case of out
   of bounds indices. Defaults to nullptr, which will instead access the first
   value from *data.
   */
  __host__ __device__ Tdata atWithDefaultPointer(
      Tidx y, Tidx x, const Tdata *defaultValuePtr = nullptr) const {
    // Prevent nullptr access by defaulting to start of data
    const Tdata *ptr =
        defaultValuePtr == nullptr ? this->data : defaultValuePtr;
    return rowIsValid(y) && colIsValid(x) ? row(internalRow(y))[internalCol(x)]
                                          : *ptr;
  }
};

/**
 * @brief A holder container that uses thrust::device_vector for an Image.
 * Use this for RAII of a device_vector that allows you to easily return an
 * Image struct for kernel calls.
 *
 * @tparam Tdata Type of Image data
 * @tparam Tidx Type of Image indices
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
template <typename Tdata, typename Tidx = int> struct DeviceImageStorage {
  thrust::device_vector<Tdata> vec;
  Tidx width;
  Tidx height;

  DeviceImageStorage() : width(0), height(0) {}
  DeviceImageStorage(const Tidx _width, const Tidx _height)
      : vec(_width * _height), width(_width), height(_height) {}

  /**
   * @brief Resizes the underlying device_vector to the specified dimensions
   * and updates the internal width/height tracking;
   * implicitly calls the .resize() so its effects are identical.
   *
   * @param _width New width
   * @param _height New height
   */
  void resize(const Tidx _width, const Tidx _height) {
    vec.resize(_width * _height);
    width = _width;
    height = _height;
  }

  /**
   * @brief Primary useful method. Returns a new Image struct that encloses the
   * pointer alone, allowing it to be passed to a kernel by value.
   *
   * @return Image struct
   */
  Image<Tdata, Tidx> image() {
    return Image<Tdata, Tidx>(vec.data().get(), width, height);
  }

  /**
   * @brief Primary useful method. Returns a new const Image struct that
   * encloses the pointer alone, allowing it to be passed to a kernel by value.
   *
   * @return Const Image struct
   */
  Image<const Tdata, Tidx> cimage() const {
    return Image<const Tdata, Tidx>(vec.data().get(), width, height);
  }
};

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
  StreamOrderedDeviceStorage() = delete;
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
#ifndef NDEBUG
    printf("StreamOrderedDeviceStorage: cudaMallocAsync of %zu bytes\n",
           m_size);
#endif
    cudaError_t err = cudaMallocAsync(&m_data, sizeof(T) * m_size, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "cudaMallocAsync failed with size " +
          std::to_string(m_size * sizeof(T)) + " and stream " +
          std::to_string(reinterpret_cast<uint64_t>(stream)));
    }
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

  // No copy/move semantics whatsoever
  StreamOrderedDeviceStorage(const StreamOrderedDeviceStorage &) = delete;
  StreamOrderedDeviceStorage(StreamOrderedDeviceStorage &&) = delete;
  StreamOrderedDeviceStorage &
  operator=(const StreamOrderedDeviceStorage &) = delete;
  StreamOrderedDeviceStorage &operator=(StreamOrderedDeviceStorage &&) = delete;

  const T *data() const { return m_data; }
  T *data() { return m_data; }
  size_t size() const { return m_size; }
  size_t capacity() const { return m_capacity; }
  cudaStream_t stream() const { return m_stream; }

protected:
  size_t m_capacity;
  size_t m_size;
  T *m_data;
  cudaStream_t m_stream;
};

template <typename Tdata, typename Tidx = int>
class StreamOrderedDeviceImageStorage
    : public StreamOrderedDeviceStorage<Tdata> {
public:
  StreamOrderedDeviceImageStorage() = delete;
  StreamOrderedDeviceImageStorage(const Tidx width, const Tidx height,
                                  const cudaStream_t stream)
      : StreamOrderedDeviceStorage<Tdata>((size_t)width * (size_t)height,
                                          stream),
        m_width(width), m_height(height) {}

  // No copy/move semantics whatsoever
  StreamOrderedDeviceImageStorage(const StreamOrderedDeviceImageStorage &) =
      delete;
  StreamOrderedDeviceImageStorage(StreamOrderedDeviceImageStorage &&) = delete;
  StreamOrderedDeviceImageStorage &
  operator=(const StreamOrderedDeviceImageStorage &) = delete;
  StreamOrderedDeviceImageStorage &
  operator=(StreamOrderedDeviceImageStorage &&) = delete;

  Tidx width() const { return m_width; }
  Tidx height() const { return m_height; }

  Image<Tdata, Tidx> image() {
    return Image<Tdata, Tidx>(this->m_data, m_width, m_height);
  }

  Image<const Tdata, Tidx> cimage() const {
    return Image<const Tdata, Tidx>(this->m_data, m_width, m_height);
  }

protected:
  Tidx m_width;
  Tidx m_height;
};

} // namespace containers

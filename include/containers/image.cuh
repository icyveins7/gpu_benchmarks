#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "stream_ordered_storage.cuh"

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
 * @param elementsPerRow Number of elements per row, including padding
 */
template <typename Tdata, typename Tidx = int> struct Image {
  static_assert(std::is_integral_v<Tidx>, "Tidx must be an integer type");

  Tdata *data;
  Tidx width;
  Tidx height;
  Tidx elementsPerRow; // includes possible padding

  /**
   * @brief Constructor. Automatically sets bytePerRow, assuming no padding.
   * Otherwise, list-initialization for the members can/should be used directly.
   */
  __host__ __device__ Image(Tdata *_data, const Tidx _width, const Tidx _height)
      : data(_data), width(_width), height(_height), elementsPerRow(width) {}

  /**
   * @brief Does the same thing as the constructor. Useful for delayed
   * initialization, like in conditional blocks.
   */
  __host__ __device__ void initialize(Tdata *_data, const Tidx _width,
                                      const Tidx _height) {
    data = _data;
    width = _width;
    height = _height;
    elementsPerRow = width;
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
    // we make a cast to int64_t for 'large' y or elementsPerRow e.g. > 2^16,
    // since this would overflow an int32
    return &data[(int64_t)y * elementsPerRow];
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

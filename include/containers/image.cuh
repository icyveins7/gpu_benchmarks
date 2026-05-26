#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "pinnedalloc.cuh"
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

  /**
   * @brief Copies the entire device image to a host image storage of the same
   * dimensions. Both source and destination must have matching width and
   * height. See PinnedHostImageStorage::toDevice() for the reverse.
   *
   * @tparam Thoststorage Host storage type (e.g. PinnedHostImageStorage)
   * @param dst Destination host storage; must have the same width and height
   * @param stream CUDA stream for the async copy
   */
  template <typename Thoststorage>
  void toHost(Thoststorage &dst, cudaStream_t stream = 0) const {
    // Check that destination matches exactly
    if (dst.width != this->width || dst.height != this->height) {
      throw std::runtime_error(
          "DeviceImageStorage::toHost: Destination size mismatch.");
    }

    cudaError_t err = cudaMemcpyAsync(
        dst.vec.data().get(), this->vec.data().get(),
        (size_t)width * height * sizeof(Tdata), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      char const *errstr = cudaGetErrorString(err);
      throw std::runtime_error("DeviceImageStorage::toHost: "
                               "cudaMemcpyAsync failed with error " +
                               std::string(errstr));
    }
  }

  /**
   * @brief Copies the entire device image into a sub-region (ROI) of a larger
   * host image storage. The device storage dimensions must match numRows x
   * numCols exactly. In typical usage, the host pinned memory is the larger
   * image, and the device memory is the carved-out ROI / tile.
   * See PinnedHostImageStorage::toDeviceFromROI() for the reverse.
   *
   * @tparam Thoststorage Host storage type (e.g. PinnedHostImageStorage)
   * @param dst Destination host storage (the larger image)
   * @param dstStartRow Starting row in the destination to write into
   * @param dstStartCol Starting column in the destination to write into
   * @param numRows Number of rows to copy (must match this->height)
   * @param numCols Number of columns to copy (must match this->width)
   * @param stream CUDA stream for the async copy
   */
  template <typename Thoststorage>
  void toHostROI(Thoststorage &dst, Tidx dstStartRow, Tidx dstStartCol,
                 Tidx numRows, Tidx numCols, cudaStream_t stream = 0) const {
    // Check that source matches the requested sub-region
    if (this->width != numCols || this->height != numRows) {
      throw std::runtime_error(
          "DeviceImageStorage::toHost: Source size mismatch. "
          "Source must match exactly the requested sub-region.");
    }
    // Bounds check on the destination
    if (dstStartRow < 0 || dstStartCol < 0 ||
        dstStartRow + numRows > dst.height ||
        dstStartCol + numCols > dst.width) {
      throw std::runtime_error("DeviceImageStorage::toHost: Sub-region "
                               "exceeds destination bounds.");
    }

    Tdata *dstPtr =
        dst.vec.data().get() + (size_t)dstStartRow * dst.width + dstStartCol;
    const Tdata *src = this->vec.data().get();

    cudaError_t err = cudaMemcpy2DAsync(
        dstPtr, dst.width * sizeof(Tdata), src, numCols * sizeof(Tdata),
        numCols * sizeof(Tdata), numRows, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      char const *errstr = cudaGetErrorString(err);
      throw std::runtime_error("DeviceImageStorage::toHost: "
                               "cudaMemcpy2DAsync failed with error " +
                               std::string(errstr));
    }
  }
};

template <typename Tdata, typename Tidx = int>
struct StreamOrderedDeviceImageStorage
    : public StreamOrderedDeviceStorage<Tdata> {
  Tidx width;
  Tidx height;

  StreamOrderedDeviceImageStorage() = delete;
  StreamOrderedDeviceImageStorage(const Tidx _width, const Tidx _height,
                                  const cudaStream_t stream)
      : StreamOrderedDeviceStorage<Tdata>((size_t)_width * (size_t)_height,
                                          stream),
        width(_width), height(_height) {}

  // No copy/move semantics whatsoever
  StreamOrderedDeviceImageStorage(const StreamOrderedDeviceImageStorage &) =
      delete;
  StreamOrderedDeviceImageStorage(StreamOrderedDeviceImageStorage &&) = delete;
  StreamOrderedDeviceImageStorage &
  operator=(const StreamOrderedDeviceImageStorage &) = delete;
  StreamOrderedDeviceImageStorage &
  operator=(StreamOrderedDeviceImageStorage &&) = delete;

  Image<Tdata, Tidx> image() {
    return Image<Tdata, Tidx>(this->m_data, width, height);
  }

  Image<const Tdata, Tidx> cimage() const {
    return Image<const Tdata, Tidx>(this->m_data, width, height);
  }
};

template <typename Tdata, typename Tidx = int> struct PinnedHostImageStorage {
  thrust::pinned_host_vector<Tdata> vec;
  Tidx width;
  Tidx height;

  PinnedHostImageStorage() : width(0), height(0) {}
  PinnedHostImageStorage(const Tidx _width, const Tidx _height)
      : vec(_width * _height), width(_width), height(_height) {}

  /**
   * @brief Resizes the underlying pinned_host_vector to the specified
   * dimensions and updates the internal width/height tracking; implicitly calls
   * the .resize() so its effects are identical.
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

  /**
   * @brief Copies the entire pinned host image to a device image storage of
   * the same dimensions. Both source and destination must have matching width
   * and height.
   * See DeviceImageStorage::toHost() for the reverse.
   *
   * @tparam Tdevicestorage Device storage type (e.g. DeviceImageStorage)
   * @param dst Destination device storage; must have the same width and height
   * @param stream CUDA stream for the async copy
   */
  template <typename Tdevicestorage>
  void toDevice(Tdevicestorage &dst, cudaStream_t stream = 0) const {
    // Check that destination matches exactly
    if (dst.width != this->width || dst.height != this->height) {
      throw std::runtime_error(
          "PinnedHostImageStorage::toDevice: Destination size mismatch.");
    }

    cudaError_t err = cudaMemcpyAsync(
        dst.vec.data().get(), this->vec.data().get(),
        (size_t)width * height * sizeof(Tdata), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      char const *errstr = cudaGetErrorString(err);
      throw std::runtime_error("PinnedHostImageStorage::toDevice: "
                               "cudaMemcpyAsync failed with error " +
                               std::string(errstr));
    }
  }

  /**
   * @brief Copies a sub-region (ROI) of the pinned host image to a device
   * image storage. The destination dimensions must match numRows x numCols
   * exactly. In typical usage, the host pinned memory is the larger image, and
   * the device memory is the carved-out ROI / tile.
   * See DeviceImageStorage::toHostROI() for the reverse.
   *
   * @tparam Tdevicestorage Device storage type (e.g. DeviceImageStorage)
   * @param dst Destination device storage (the tile); must match numRows x
   * numCols
   * @param srcStartRow Starting row in the source (this) to read from
   * @param srcStartCol Starting column in the source (this) to read from
   * @param numRows Number of rows to copy (must match dst.height)
   * @param numCols Number of columns to copy (must match dst.width)
   * @param stream CUDA stream for the async copy
   */
  template <typename Tdevicestorage>
  void toDeviceFromROI(Tdevicestorage &dst, Tidx srcStartRow, Tidx srcStartCol,
                       Tidx numRows, Tidx numCols,
                       cudaStream_t stream = 0) const {
    // Check that destination matches the requested sub-region
    if (dst.width != numCols || dst.height != numRows) {
      throw std::runtime_error(
          "PinnedHostImageStorage::toDevice: Destination size mismatch. "
          "Destination must match exactly; resize first if required.");
    }
    // Bounds check on the source
    if (srcStartRow < 0 || srcStartCol < 0 || srcStartRow + numRows > height ||
        srcStartCol + numCols > width) {
      throw std::runtime_error("PinnedHostImageStorage::toDevice: Sub-region "
                               "exceeds source bounds.");
    }

    const Tdata *src =
        this->vec.data().get() + (size_t)srcStartRow * width + srcStartCol;
    Tdata *dstPtr = dst.vec.data().get();

    cudaError_t err = cudaMemcpy2DAsync(
        dstPtr, numCols * sizeof(Tdata), src, width * sizeof(Tdata),
        numCols * sizeof(Tdata), numRows, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      char const *errstr = cudaGetErrorString(err);
      throw std::runtime_error("PinnedHostImageStorage::toDevice: "
                               "cudaMemcpy2DAsync failed with error " +
                               std::string(errstr));
    }
  }
};

} // namespace containers

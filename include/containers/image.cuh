#include <cstdint>

namespace containers {

template <typename Tdata, typename Tidx = unsigned int> struct Image {
  Tdata *data;
  Tidx width;
  Tidx height;
  Tidx bytesPerRow;

  /**
   * @brief Constructor. Automatically sets bytePerRow, assuming no padding.
   * Otherwise, list-initialization for the members can/should be used directly.
   */
  __host__ __device__ Image(Tdata *_data, Tidx _width, Tidx _height)
      : data(_data), width(_width), height(_height),
        bytesPerRow(width * sizeof(Tdata)) {}

  __host__ __device__ bool rowIsValid(Tidx y) const { return y < height; }
  __host__ __device__ bool colIsValid(Tidx x) const { return x < width; }

  /**
   * @brief Returns a pointer to the specified row.
   */
  __host__ __device__ Tdata *row(Tidx y) const {
    return reinterpret_cast<Tdata *>(reinterpret_cast<uint8_t *>(data) +
                                     y * bytesPerRow);
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

} // namespace containers

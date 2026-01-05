#include <cstdint>

namespace containers {

template <typename Tdata, typename Tidx = int> struct Image {
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

  __host__ __device__ bool rowIsValid(Tidx y) { return y < height; }
  __host__ __device__ bool colIsValid(Tidx x) { return x < width; }

  /**
   * @brief Returns a pointer to the specified row.
   */
  __host__ __device__ Tdata *row(Tidx y) {
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
};

} // namespace containers

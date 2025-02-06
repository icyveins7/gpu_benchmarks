#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <cmath>
#include <random>
#include "../include/sharedmem.cuh"

#include <stdexcept>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

template <typename T>
void quickView(const thrust::host_vector<T> &v, const size_t width,
               const size_t height) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      printf("%.2f ", (float)v[i * width + j]);
    }
    printf("\n");
  }
}

template <typename T>
inline __device__ T bilinearInterpolate(
  const T topLeft, const T topRight, const T btmLeft, const T btmRight,
  const float x, const float y
){
  // Interpolate on the rows first
  const T top = topLeft + (topRight - topLeft) * (x - floorf(x));
  const T btm = btmLeft + (btmRight - btmLeft) * (x - floorf(x));
  // Then interpolate the two row interpolants
  const T result = top + (btm - top) * (y - floorf(y));
  return result;
}

template <typename T>
inline T equivalentBilinearInterpolate(
  const T topLeft, const T topRight, const T btmLeft, const T btmRight,
  const float x, const float y
){
  // Interpolate on the rows first
  const T top = topLeft + (topRight - topLeft) * (x - floorf(x));
  const T btm = btmLeft + (btmRight - btmLeft) * (x - floorf(x));
  // Then interpolate the two row interpolants
  const T result = top + (btm - top) * (y - floorf(y));
  return result;
}


/**
 * @brief Retrieves the pixel index while performing bounds checking.
 *        This prevents pixels outside the image bounds from being accessed.
 *
 * @param xIdx Column index
 * @param yIdx Row index
 * @param srcWidth Number of columns in pixels
 * @param srcHeight Number of rows in pixels
 * @param idx1d Output 1D index, only to be used if return value is true
 * @return True if the pixel is within bounds
 */
inline __device__ static bool getPixelIdxAsIf1D(
  const size_t xIdx, const size_t yIdx,
  const size_t srcWidth, const size_t srcHeight,
  size_t& idx1d
){
  if (xIdx > srcWidth || yIdx >= srcHeight)
    return false;
  idx1d = yIdx * srcWidth + xIdx;
  return true;
}

inline static bool getPixelIdxAsIf1D_host(
  const size_t xIdx, const size_t yIdx,
  const size_t srcWidth, const size_t srcHeight,
  size_t& idx1d
){
  // This is identical to above, just for use in host-side checking.
  if (xIdx > srcWidth || yIdx >= srcHeight)
    return false;
  idx1d = yIdx * srcWidth + xIdx;
  return true;
};

/**
 * @brief A naive bilinear interpolation kernel.
 *        Assumes both source and output pixels are the same type.
 *        Interpolation is assumed to be done using single-precision floats.
 *        Assumes source and output images are contiguous in memory; that means
 *        there is no extra padding at the end of rows (they are both effectively 1D
 *        long arrays).
 *
 *        Explicit checks are made to read/write to the requested input/output pixel addresses,
 *        as well as the source image addresses.
 *
 *        Concretely, this kernel outputs a value only when the following conditions are met:
 *        1) x is within [0, srcWidth - 1], including the edge itself
 *        2) y is within [0, srcHeight - 1], including the edge itself
 *
 *        In any other scenario, the kernel should not write an output for the pixel i.e.
 *        output pixel memory is untouched.
 *
 * @tparam T Underlying type of source pixels and output pixels.
 * @param d_src Source image pointer. Assumes contiguous memory (no padding in a row).
 * @param srcWidth Source width in pixels i.e. number of columns
 * @param srcHeight Source height in pixels i.e. number of rows
 * @param d_x Requested x pixel locations pointer (has dimensions reqWidth x reqHeight)
 * @param d_y Requested y pixel locations pointer (has dimensions reqWidth x reqHeight)
 * @param reqWidth Requested output image width in pixels i.e. number of columns
 * @param reqHeight Requested output image height in pixels i.e. number of rows
 * @param d_out Output image pointer.
 * @return 
 */
template <typename T>
__global__ void naiveRemap(
  const T* __restrict__ d_src,
  const size_t srcWidth,
  const size_t srcHeight,
  const float* __restrict__ d_x,
  const float* __restrict__ d_y,
  const size_t reqWidth,
  const size_t reqHeight,
  T* __restrict__ d_out
){
  const int xt = threadIdx.x + blockDim.x * blockIdx.x;
  const int yt = threadIdx.y + blockDim.y * blockIdx.y;
  // This index is used for the requested x, y and the output
  const int idx = reqWidth * yt + xt;

  // Explicit check for out of bounds for x, y and output
  if (idx < reqWidth * reqHeight) {
    // Retrieve the requested pixel locations
    const float x = d_x[idx];
    const float y = d_y[idx];

    // Ignore if the requested point itself is outside source
    // NOTE: the -1 is important!
    if (x < 0 || x > srcWidth - 1 || y < 0 || y > srcHeight - 1)
      return;

    // Get the 4 corner indices
    const size_t xiTopLeft = static_cast<size_t>(floorf(x));
    const size_t yiTopLeft = static_cast<size_t>(floorf(y));

    size_t iTopLeft, iTopRight, iBtmLeft, iBtmRight;

    // Allocate values for the 4 corners
    float topLeft, topRight, btmLeft, btmRight;

    // NOTE: In the following cases, we can always set the value of the corner pixel to be 0
    // whenever it is an invalid index. The reason why we can do this is because we have already
    // taken care of the cases where the requested pixel is out of bounds. Hence the following
    // section of code only executes for those in-bounds (including pixels collinear with the edges).
    // We now write values of 0 for these 'invalid' pixels so that the processing of pixels
    // collinear with the edges will be correct.

    if (!getPixelIdxAsIf1D(xiTopLeft, yiTopLeft, srcWidth, srcHeight, iTopLeft))
      topLeft = 0;
    else
      topLeft = d_src[iTopLeft];

    if (!getPixelIdxAsIf1D(xiTopLeft + 1, yiTopLeft, srcWidth, srcHeight, iTopRight))
      topRight = 0;
    else
      topRight = d_src[iTopRight];

    if (!getPixelIdxAsIf1D(xiTopLeft, yiTopLeft + 1, srcWidth, srcHeight, iBtmLeft))
      btmLeft = 0;
    else
      btmLeft = d_src[iBtmLeft];

    if (!getPixelIdxAsIf1D(xiTopLeft + 1, yiTopLeft + 1, srcWidth, srcHeight, iBtmRight))
      btmRight = 0;
    else
      btmRight = d_src[iBtmRight];

    // Interpolate values
    const float result = bilinearInterpolate(topLeft, topRight, btmLeft, btmRight, x, y);

    // Write result as destination type
    d_out[idx] = (T)result;
  }
}

/**
 * @brief Parent class for remapping. Should not be instantiated directly.
 *
 * @tparam T Type of source and output images
 * @param height Source image height in pixels
 * @param width Source image width in pixels
 * @return 
 */
template <typename T>
class Remap
{
public:
  /**
   * @brief Parent class constructor to generate a sequence of values as the source image.
   *
   * @param height Source image height in pixels
   * @param width Source image width in pixels
   */
  Remap(const size_t height, const size_t width) : m_srcHeight(height), m_srcWidth(width) {
    printf("Filling device src values..\n");
    // Generate incremental values
    m_h_src.resize(height * width);
    m_d_src.resize(height * width);
    thrust::sequence(m_d_src.begin(), m_d_src.end(), 0);
    printf("Copying to host..\n");
    m_h_src = m_d_src;
  }

  /**
   * @brief Parent class constructor to copy an existing host vector as the source image.
   *
   * @param h_src Existing thrust::host_vector to use as source image
   * @param height Source image height in pixels
   * @param width Source image width in pixels
   */
  Remap(const thrust::host_vector<T>& h_src, const size_t height, const size_t width)
    : m_h_src(h_src), m_srcHeight(height), m_srcWidth(width)
  {
    // Copy to device directly
    m_d_src.resize(height * width);
    m_d_src = m_h_src;
  }

  /**
   * @brief Sets the threads per block (per dimension) for kernel calls.
   *        Kernel calls are assumed to use 2D blocks of (TPB, TPB)
   *
   * @param THREADS_PER_BLK 
   */
  void set_tpb(const int THREADS_PER_BLK) { m_THREADS_PER_BLK = THREADS_PER_BLK; }

  // Getters
  thrust::device_vector<T>& get_d_src() { return m_d_src; }
  thrust::host_vector<T>& get_h_src() { return m_h_src; }

  // Placeholders for actual computations to do in child classes

  /**
   * @brief Host-side runner to perform remapping. This is primarily used to validate the kernel.
   *
   * @param h_x Host vector of requested x locations
   * @param h_y Host vector of requested y locations
   * @param reqWidth Requested locations width in pixels
   * @param reqHeight Requested locations height in pixels
   * @param h_out Output host vector
   */
  void h_run(const thrust::host_vector<float>& h_x,
             const thrust::host_vector<float>& h_y,
             const size_t reqWidth, const size_t reqHeight,
             thrust::host_vector<T>& h_out) 
  {
    for (size_t i = 0; i < reqHeight; ++i)
    {
      for (size_t j = 0; j < reqWidth; ++j)
      {
        const size_t idx = i * reqWidth + j;
        // Retrieve the x and y requested pixel locations
        const float x = h_x[idx];
        const float y = h_y[idx];

        // Ignore if the requested point itself is outside
        // NOTE: the -1 is important!
        if (x < 0 || x > this->m_srcWidth - 1 || y < 0 || y > this->m_srcHeight - 1)
          continue;

        // Get the 4 corner indices
        const size_t xiTopLeft = static_cast<size_t>(floorf(x));
        const size_t yiTopLeft = static_cast<size_t>(floorf(y));

        size_t iTopLeft, iTopRight, iBtmLeft, iBtmRight;

        // Allocate values for the 4 corners
        float topLeft, topRight, btmLeft, btmRight;

        // NOTE: In the following cases, we can always set the value of the corner pixel to be 0
        // whenever it is an invalid index. The reason why we can do this is because we have already
        // taken care of the cases where the requested pixel is out of bounds. Hence the following
        // section of code only executes for those in-bounds (including pixels collinear with the edges).
        // We now write values of 0 for these 'invalid' pixels so that the processing of pixels
        // collinear with the edges will be correct.

        if (!getPixelIdxAsIf1D_host(xiTopLeft, yiTopLeft, this->m_srcWidth, this->m_srcHeight, iTopLeft))
          topLeft = 0;
        else
          topLeft = m_h_src[iTopLeft];

        if (!getPixelIdxAsIf1D_host(xiTopLeft + 1, yiTopLeft, this->m_srcWidth, this->m_srcHeight, iTopRight))
          topRight = 0;
        else
          topRight = m_h_src[iTopRight];

        if (!getPixelIdxAsIf1D_host(xiTopLeft, yiTopLeft + 1, this->m_srcWidth, this->m_srcHeight, iBtmLeft))
          btmLeft = 0;
        else
          btmLeft = m_h_src[iBtmLeft];

        if (!getPixelIdxAsIf1D_host(xiTopLeft + 1, yiTopLeft + 1, this->m_srcWidth, this->m_srcHeight, iBtmRight))
          btmRight = 0;
        else
          btmRight = m_h_src[iBtmRight];

        // Interpolate values
        const float result = equivalentBilinearInterpolate(topLeft, topRight, btmLeft, btmRight, x, y);

        h_out[idx] = result;
      }
    }
  };

  /**
   * @brief Placeholder device-side runner to perform remapping.
   *
   * @param d_x Device vector of requested x locations
   * @param d_y Device vector of requested y locations
   * @param reqWidth Requested locations width in pixels
   * @param reqHeight Requested locations height in pixels
   * @param d_out Output device vector
   */
  void d_run(const thrust::device_vector<T>& d_x,
             const thrust::device_vector<T>& d_y,
             const size_t reqWidth, const size_t reqHeight,
             thrust::device_vector<T>& d_out) {};

protected:
  size_t m_srcHeight = 0;
  size_t m_srcWidth = 0;
  thrust::host_vector<T> m_h_src;
  thrust::device_vector<T> m_d_src;
  int m_THREADS_PER_BLK = 16;
};

template <typename T>
class NaiveRemap : public Remap<T>
{
public:
  NaiveRemap(const size_t height, const size_t width) : Remap<T>(height, width) {}
  NaiveRemap(const thrust::host_vector<T>& h_src, const size_t height, const size_t width) : Remap<T>(h_src, height, width) {}

  void d_run(const thrust::device_vector<float>& d_x,
             const thrust::device_vector<float>& d_y,
             const size_t reqWidth, const size_t reqHeight,
             thrust::device_vector<T>& d_out) 
  {
    int numBlks_x = static_cast<int>(reqWidth / this->m_THREADS_PER_BLK) + 1;
    int numBlks_y = static_cast<int>(reqHeight / this->m_THREADS_PER_BLK) + 1;
    printf("Using grid dims = %d, %d\n", numBlks_x, numBlks_y);
    dim3 dimGrid(numBlks_x, numBlks_y);
    printf("Using blk dims = %d, %d\n", this->m_THREADS_PER_BLK, this->m_THREADS_PER_BLK);
    dim3 dimBlk(this->m_THREADS_PER_BLK, this->m_THREADS_PER_BLK);
    // Execute kernel
    naiveRemap<<<dimGrid, dimBlk>>>(
      thrust::raw_pointer_cast(this->m_d_src.data()),
      this->m_srcWidth,
      this->m_srcHeight,
      thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()),
      reqWidth,
      reqHeight,
      thrust::raw_pointer_cast(d_out.data())
    );
  }
};


/**
 * @brief Retrieves the pixel index while performing bounds checking.
 *        This is slightly different as it simply checks if the access is within
 *        the unwrapped 1-D data bounds, instead of the image width and height
 *        individually.
 *
 * @param xIdx Column index
 * @param yIdx Row index
 * @param srcWidth Number of columns in pixels
 * @param srcHeight Number of rows in pixels
 * @param idx1d Output 1D index, only to be used if return value is true
 * @return True if the pixel is within bounds
 */
inline __device__ static bool getUnwrappedPixelIdxAsIf1D(
  const size_t xUnwrapIdx, const size_t yUnwrapIdx,
  const size_t srcWidth, const size_t srcHeight,
  size_t& idx1d
){
  idx1d = yUnwrapIdx * srcWidth + xUnwrapIdx;
  if (idx1d >= srcWidth * srcHeight)
    return false;
  return true;
}

/**
 * @brief This is similar to the naiveRemap kernel, but out-of-bounds accesses
 *        now wrap-around and produce valid output, as long as the requested pixel
 *        is still within the unwrapped length of the image.
 *
 *        Concretely, this expands valid requests to 'right' of the source image, with
 *        certain caveats.
 *
 *        Example with source pixels A-F:
 *
 *        A    B
 *               * -> this request is ok, and will use the square BCED
 *        C    D
 *               * -> this request is not ok as the bottom right does not exist
 *        E    F
 *
 *        The first request treats the image as if it were like this:
 *
 *        A    B    C
 *               *
 *        C    D    E ...
 *
 *        Hence the only requirement becomes that of existence of the wrapped-around
 *        corner pixels; it cannot be too far as to extend beyond the end of the image data.
 *
 *        Extra care must be taken to handle the bottom edge case that occurred in the
 *        original naive kernel, as these pixels would have 2 'invalid' corners but
 *        should still be processed:
 *
 *        A    B
 *
 *        C  * D
 *           |
 *           -> this request is still ok!
 *
 * @tparam T Underlying type of source pixels and output pixels.
 * @param d_src Source image pointer. Assumes contiguous memory (no padding in a row).
 * @param srcWidth Source width in pixels i.e. number of columns
 * @param srcHeight Source height in pixels i.e. number of rows
 * @param d_x Requested x pixel locations pointer (has dimensions reqWidth x reqHeight)
 * @param d_y Requested y pixel locations pointer (has dimensions reqWidth x reqHeight)
 * @param reqWidth Requested output image width in pixels i.e. number of columns
 * @param reqHeight Requested output image height in pixels i.e. number of rows
 * @param d_out Output image pointer.
 * @return 
 */
template <typename T>
__global__ void naiveRemapWraparound(
  const T* __restrict__ d_src,
  const size_t srcWidth,
  const size_t srcHeight,
  const float* __restrict__ d_x,
  const float* __restrict__ d_y,
  const size_t reqWidth,
  const size_t reqHeight,
  T* __restrict__ d_out
){
  const int xt = threadIdx.x + blockDim.x * blockIdx.x;
  const int yt = threadIdx.y + blockDim.y * blockIdx.y;
  // This index is used for the requested x, y and the output
  const int idx = reqWidth * yt + xt;

  // Explicit check for out of bounds for x, y and output
  if (idx < reqWidth * reqHeight) {
    // Retrieve the requested pixel locations
    const float x = d_x[idx];
    const float y = d_y[idx];

    // The boundary requirements are now based on the unwrapped position
    // Assume that width = M - 1, height = N - 1
    // First, we unwrap x by checking how far it is over the edge M (NOTE: not M - 1)
    int yIncrements = 0;
    const float xUnwrap = remquof(x, srcWidth, &yIncrements);
    const float yUnwrap = y + static_cast<float>(yIncrements);

    // The source image can now effectively be treated as an (M+1)x(N) image,
    // where the last column has only N-1 pixels. That is, the last column has the same values
    // as the first column, with one row shifted up:
    //
    //         0    1    2  .... M-1  M
    //      |  ------------------------
    //   0  |  A    B    C  .... D    E
    //   1  |  E    F    G  .... H    I
    //      |  .......................
    // N-2  |  P    Q    R  .... S    T
    // N-1  |  T    U    V  .... W    ?
    //
    // All the previous rules for validity of requested pixels now apply to the unwrapped values
    // in the same way they would have for the original naiveRemap, except that the image is now
    // (M+1)x(N) like this.
    // The only special case is the square at the bottom right, which lacks a valid value;
    // we can simply check that unwrapped pixel coordinates in this square are not processed.
    // NOTE: the final column is technically non-inclusive i.e. [0, M)

    // Check 1. Within the (M+1)x(N)
    if (yUnwrap < 0 || yUnwrap > srcHeight - 1 || xUnwrap < 0 || xUnwrap >= srcWidth) // now x is not -1!
      return;

    // Check 2. Exclude the final square (need to include the top and left edge itself)
    if (xUnwrap > srcWidth - 1 && yUnwrap > srcHeight - 2)
      return;

    // If we reach here, then we are safe to evaluate all 4 corner pixels;
    // any pixel that is unreachable in memory can just be set to 0, the math will take care of the rest,
    // same as in naiveRemap

    // Get the 4 corner indices using unwrapped coordinates now
    const size_t xiTopLeft = static_cast<size_t>(xUnwrap);
    const size_t yiTopLeft = static_cast<size_t>(yUnwrap);

    size_t iTopLeft, iTopRight, iBtmLeft, iBtmRight;

    // Allocate values for the 4 corners
    float topLeft, topRight, btmLeft, btmRight;

    if (!getPixelIdxAsIf1D(xiTopLeft, yiTopLeft, srcWidth, srcHeight, iTopLeft))
      topLeft = 0;
    else
      topLeft = d_src[iTopLeft];

    if (!getPixelIdxAsIf1D(xiTopLeft + 1, yiTopLeft, srcWidth, srcHeight, iTopRight))
      topRight = 0;
    else
      topRight = d_src[iTopRight];

    if (!getPixelIdxAsIf1D(xiTopLeft, yiTopLeft + 1, srcWidth, srcHeight, iBtmLeft))
      btmLeft = 0;
    else
      btmLeft = d_src[iBtmLeft];

    if (!getPixelIdxAsIf1D(xiTopLeft + 1, yiTopLeft + 1, srcWidth, srcHeight, iBtmRight))
      btmRight = 0;
    else
      btmRight = d_src[iBtmRight];

    // Interpolate values
    const float result = bilinearInterpolate(topLeft, topRight, btmLeft, btmRight, x, y);

    // Write result as destination type
    d_out[idx] = (T)result;
  }
}

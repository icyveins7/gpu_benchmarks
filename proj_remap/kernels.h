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
    const size_t iTopLeft = (size_t)floorf(y) * srcWidth + (size_t)floorf(x);
    const size_t iTopRight = iTopLeft + 1;
    const size_t iBtmLeft = iTopLeft + srcWidth;
    const size_t iBtmRight = iTopRight + srcWidth;

    // End if any of them are out of bounds of source
    size_t srcSize = srcWidth * srcHeight;
    if (
      iTopLeft >= srcSize || // it should be sufficient to only check top left and btm right
      // iTopRight >= srcSize ||
      // iBtmLeft >= srcSize ||
      iBtmRight >= srcSize
    )
      return;

    // Retrieve the 4 corner values and cast to floating point
    const float topLeft = d_src[iTopLeft];
    const float topRight = d_src[iTopRight];
    const float btmLeft = d_src[iBtmLeft];
    const float btmRight = d_src[iBtmRight];

    // Interpolate values
    const float result = bilinearInterpolate(topLeft, topRight, btmLeft, btmRight, x, y);

    // Write result as destination type
    d_out[idx] = (T)result;
  }
}

template <typename T>
class Remap
{
public:
  Remap(const size_t height, const size_t width) : m_srcHeight(height), m_srcWidth(width) {
    printf("Filling device src values..\n");
    // Generate incremental values
    m_h_src.resize(height * width);
    m_d_src.resize(height * width);
    thrust::sequence(m_d_src.begin(), m_d_src.end(), 0);
    printf("Copying to host..\n");
    m_h_src = m_d_src;

    // quickView(m_h_src, width, height);

  }
  Remap(const thrust::host_vector<T>& h_src, const size_t height, const size_t width)
    : m_h_src(h_src), m_srcHeight(height), m_srcWidth(width)
  {
    // Copy to device directly
    m_d_src.resize(height * width);
    m_d_src = m_h_src;
  }

  void set_tpb(const int THREADS_PER_BLK) { m_THREADS_PER_BLK = THREADS_PER_BLK; }

  // Getters
  const thrust::device_vector<T>& get_d_src() { return m_d_src; }
  const thrust::host_vector<T>& get_h_src() { return m_h_src; }

  // Placeholders for actual computations to do in child classes
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

        // Retrieve the surrounding pixels
        const size_t iTopLeft = (size_t)floorf(y) * this->m_srcWidth + (size_t)floorf(x);
        const size_t iTopRight = iTopLeft + 1;
        const size_t iBtmLeft = iTopLeft + this->m_srcWidth;
        const size_t iBtmRight = iTopRight + this->m_srcWidth;

        if (
          iTopLeft >= m_h_src.size() ||
          iTopRight >= m_h_src.size() ||
          iBtmLeft >= m_h_src.size() ||
          iBtmRight >= m_h_src.size()
        )
          continue;

        const float topLeft = m_h_src[iTopLeft];
        const float topRight = m_h_src[iTopRight];
        const float btmLeft = m_h_src[iBtmLeft];
        const float btmRight = m_h_src[iBtmRight];


        // Interpolate values
        const float result = equivalentBilinearInterpolate(topLeft, topRight, btmLeft, btmRight, x, y);

        h_out[idx] = result;
      }
    }
  };

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
  NaiveRemap(const thrust::host_vector<T>& h_src, const size_t height, const size_t width) : Remap<T>(h_src) {}

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

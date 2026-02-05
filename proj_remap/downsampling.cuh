#pragma once

#include <iostream>
#include <type_traits>

template <typename Tidx = int> struct CentreAlignedDownsampler {
  static_assert(std::is_integral<Tidx>::value, "Tidx must be integral");

  CentreAlignedDownsampler() {}
  CentreAlignedDownsampler(const Tidx _inWidth, const Tidx _inHeight,
                           const Tidx _xRatio, const Tidx _yRatio)
      : inWidth(_inWidth), inHeight(_inHeight), cx(_inWidth / 2),
        cy(_inHeight / 2), xRatio(_xRatio), yRatio(_yRatio) {
    // Determine start/end indices by spreading
    // as equally as possible in both directions,
    // without exceeding the first/last pixel
    Tidx xStepsBefore = (double)cx / (double)xRatio;
    Tidx xStepsAfter = (double)(_inWidth - 1 - cx) / (double)(xRatio);
    Tidx yStepsBefore = (double)cy / (double)yRatio;
    Tidx yStepsAfter = (double)(_inHeight - 1 - cy) / (double)(yRatio);

    this->xStartIdx = cx - xStepsBefore * xRatio;
    this->xEndIdx = cx + xStepsAfter * xRatio;
    this->yStartIdx = cy - yStepsBefore * yRatio;
    this->yEndIdx = cy + yStepsAfter * yRatio;

    printf("%d x %d, ratio %d/%d, x %d to %d, y %d to %d\n", _inWidth,
           _inHeight, xRatio, yRatio, xStartIdx, xEndIdx, yStartIdx, yEndIdx);
  }

  __host__ __device__ Tidx outputWidth() const {
    return (xEndIdx - xStartIdx) / xRatio + 1;
  }
  __host__ __device__ Tidx outputHeight() const {
    return (yEndIdx - yStartIdx) / yRatio + 1;
  }

  template <typename T>
  __host__ void downsampleImageOnHost(const T *in, T *out) {
    for (Tidx i = 0; i < outputHeight(); ++i) {
      for (Tidx j = 0; j < outputWidth(); ++j) {
        downsampleImagePixel(in, out, i, j);
      }
    }
  }

  template <typename T>
  __forceinline__ __host__ __device__ void
  downsampleImagePixel(const T *in, T *out, const Tidx row, const Tidx col) {
    Tidx in_y = this->yStartIdx + row * yRatio;
    Tidx in_x = this->xStartIdx + col * xRatio;
    out[row * outputWidth() + col] = in[in_y * inWidth + in_x];
  }

  Tidx inWidth;
  Tidx inHeight;
  Tidx cx;
  Tidx cy;
  Tidx xRatio;
  Tidx yRatio;
  Tidx xStartIdx;
  Tidx yStartIdx;
  Tidx xEndIdx;
  Tidx yEndIdx;
};

template <typename T, typename Tidx>
__global__ void
downsampleImageOnDeviceKernel(const T *in, T *out,
                              CentreAlignedDownsampler<Tidx> ds) {
  for (int i = 0; i < ds.outputHeight(); ++i) {
    for (int j = 0; j < ds.outputWidth(); ++j) {
      ds.downsampleImagePixel(in, out, i, j);
    }
  }
}

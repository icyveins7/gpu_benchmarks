#include "containers/image.cuh"
#include "manualConv.h"
#include "satsimpl.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

template <typename Tin, typename Trowsum, typename Tsat>
void validate_preprocessing(const thrust::host_vector<Tin> &h_data,
                            const thrust::host_vector<Trowsum> &h_rowSums,
                            const thrust::host_vector<Tsat> &h_sat,
                            const int width, const int height) {
  // Row sum check
  for (int i = 0; i < height; ++i) {
    Trowsum sum = 0;
    for (int j = 0; j < width; ++j) {
      sum += h_data[i * width + j];
      ASSERT_EQ(sum, h_rowSums[i * width + j]);
    }
  }
  // SATs check after row sums
  for (int j = 0; j < width; ++j) {
    Tsat sum = 0;
    for (int i = 0; i < height; ++i) {
      sum += h_rowSums[i * width + j];
      ASSERT_EQ(sum, h_sat[i * width + j]);
    }
  }
}

template <typename Tin, typename Trowsum, typename Tsat, typename Tout,
          typename Tscale>
void test_convolve_singlefilter(const Tscale *scales,
                                const double *radiusPixels, const int numDisks,
                                const int width, const int height) {
  // Construct disks and containers for them
  sats::FilterOfDisksRowSATCreator<int16_t> filter(scales, radiusPixels,
                                                   numDisks);

  // Construct sample input
  containers::DeviceImageStorage<Tin> d_data(width, height);
  thrust::host_vector<Tin> h_data(height * width);
  for (int i = 0; i < height * width; i++) {
    h_data[i] = std::rand() % 10 - 5;
  }
  thrust::copy(h_data.begin(), h_data.end(), d_data.vec.begin());

  // Construct convolver
  sats::PrefixRowsSAT<Tin, Tout, Tout> convolver(height, width);
  convolver.preprocess(d_data.vec.data().get());
  // Check preprocessing
  thrust::host_vector<Trowsum> h_rowSums = convolver.d_rowSums().vec;
  thrust::host_vector<Tsat> h_sat = convolver.d_sat().vec;
  validate_preprocessing(h_data, h_rowSums, h_sat, width, height);

  // Run kernel for convolution
  containers::DeviceImageStorage<Tout> d_out(width, height);
  {
    constexpr int factor = 1; // changing to 2 didn't make noticeable difference
    constexpr dim3 tpb(32, 16); // 32x8 or 32x16 seems better than 32x4
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    sats::convolve_via_SAT_and_rowSums_naive_kernel<Tin, Trowsum, Tsat, Tout,
                                                    unsigned int, int16_t>
        <<<blks, tpb>>>(filter.toDevice(), d_data.cimage(), //
                        convolver.d_rowSums().cimage(),     //
                        convolver.d_sat().cimage(),         //
                        d_out.image());
  }

  thrust::host_vector<Tout> h_out = d_out.vec;

  // Check final output
  std::vector<double> mat = filter.makeMat<double>();
  std::vector<double> checkOut = convExplicitly<Tin, double, double>(
      mat, filter.maxDiskLength(), h_data.data(), height, width);

  for (size_t i = 0; i < checkOut.size(); ++i) {
    ASSERT_EQ(checkOut[i], h_out[i]);
  }
}

TEST(ConvolverSAT_prefixRows_singleFilter_1disk, size100x100) {
  int width = 100;
  int height = 100;

  using Tscale = double;
  using Tin = int32_t;
  using Trowsum = int64_t;
  using Tsat = int64_t;
  using Tout = int64_t;

  const int numDisks = 1;
  Tscale scales[numDisks] = {1.0};
  double radiusPixels[numDisks] = {1.0};

  test_convolve_singlefilter<Tin, Trowsum, Tsat, Tout, Tscale>(
      scales, radiusPixels, numDisks, width, height);
}

TEST(ConvolverSAT_prefixRows_singleFilter_4disk, size100x100) {
  int width = 100;
  int height = 100;

  using Tscale = double;
  using Tin = int32_t;
  using Trowsum = int64_t;
  using Tsat = int64_t;
  using Tout = int64_t;

  const int numDisks = 4;
  Tscale scales[numDisks] = {1.0, 2.0, 3.0, 4.0};
  double radiusPixels[numDisks] = {5.0, 4.0, 3.0, 2.0};

  test_convolve_singlefilter<Tin, Trowsum, Tsat, Tout, Tscale>(
      scales, radiusPixels, numDisks, width, height);
}

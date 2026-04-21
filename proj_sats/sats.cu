#include <iostream>
#include <random>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
#include "manualConv.h"
#include "satsimpl.cuh"
#include "transpose.cuh"

// template <typename Tdata, typename Tidx, int BLOCK_THREADS>
// __device__ void computeRowPrefixSums_BlockPerRow(
//     containers::Image<Tdata, Tidx> image,
//     containers::Image<Tdata, Tidx> rowSums,
//     typename cub::BlockScan<Tdata, BLOCK_THREADS>::TempStorage &temp_storage)
//     {
//   using BlockScan = cub::BlockScan<Tdata, BLOCK_THREADS>;
//
//   // Each block does one row
//   for (int i = blockIdx.x; i < image.height; i += gridDim.x) {
//     // Define a carryover for this row
//     Tdata carryover = 0;
//
//     // Read elements in the row
//     for (int j = 0; j < image.width; j += blockDim.x) {
//       Tdata tElem = 0;           // default value
//       Tidx jt = j + threadIdx.x; // column for this thread
//       if (image.colIsValid(jt)) {
//         tElem = image.at(i, jt);
//       }
//       // CUB is great, no need to syncthreads since it performs internal warp
//       // syncs
//       Tdata newCarryover;
//       BlockScan(temp_storage)
//           .InclusiveSum(tElem, tElem, newCarryover); // in-place write
//
//       // Add carryover
//       tElem += carryover;
//
//       // Also add the new carryover
//       carryover += newCarryover;
//
//       // Write back
//       if (image.colIsValid(jt)) {
//         rowSums.at(i, jt) = tElem;
//       }
//
//       // In order to reuse everything, we need to sync
//       __syncthreads();
//     }
//   }
// }
//
// template <typename Tdata, typename Tidx, int BLOCK_THREADS>
// __global__ void computeSATkernel(containers::Image<Tdata, Tidx> image,
//                                  containers::Image<Tdata, Tidx> rowSums,
//                                  containers::Image<Tdata, Tidx> sat) {
//   using BlockScan = cub::BlockScan<Tdata, BLOCK_THREADS>;
//   using TempStorage = typename BlockScan::TempStorage;
//
//   __shared__ TempStorage temp_storage;
//
//   computeRowPrefixSums_BlockPerRow<Tdata, Tidx, BLOCK_THREADS>(image,
//   rowSums,
//                                                                temp_storage);
//
//   // TODO: do the columns
// }

int main(int argc, char **argv) {
  printf("Summed area tables\n");

  int height = 5, width = 5;
  if (argc >= 3) {
    height = atoi(argv[1]);
    width = atoi(argv[2]);
  }
  printf(" Image size (rows x columns): %d x %d\n", height, width);

  double radiusPixels[4] = {1.1, 0, 0, 0};
  double scaleList[4] = {1.0, 0, 0, 0};
  int numDisks = 1;
  if (argc >= 4) {
    for (int i = 0; i < std::min(4, argc - 3); i++) {
      radiusPixels[i] = atof(argv[3 + i]);
      scaleList[i] = i + 1;
      if (i >= 1) {
        numDisks++;
      }
    }
  }
  for (int i = 0; i < numDisks; ++i)
    printf(" Radius in pixels: %f\n", radiusPixels[i]);

  using Tin = int32_t;  // input is small
  using Tout = int64_t; // use 64-bit for everything else

  thrust::host_vector<Tin> h_data(height * width);
  for (int i = 0; i < height * width; i++) {
    h_data[i] = std::rand() % 10 - 5;
  }

  containers::DeviceImageStorage<Tin> d_data(width, height);
  thrust::copy(h_data.begin(), h_data.end(), d_data.vec.begin());

  containers::DeviceImageStorage<Tout> d_out(width, height);

  // Construct disks and containers for them
  sats::FilterOfDisksRowSATCreator<int16_t> filter(scaleList, radiusPixels,
                                                   numDisks);

  // Print lots of things to check?
  filter.print();
  std::vector<double> mat = filter.makeMat<double>();

  if (filter.maxDiskLength() < 100) {
    for (int i = 0; i < filter.maxDiskLength(); ++i) {
      for (int j = 0; j < filter.maxDiskLength(); ++j) {
        printf("%.1f ", mat[i * filter.maxDiskLength() + j]);
      }
      printf("\n");
    }
  }
  // End of debug printing

  // Construct the object to handle preprocessing
  sats::PrefixRowsSAT<Tin, Tout, Tout> preprocessor(height, width);
  preprocessor.preprocess(d_data.vec.data().get());

  // === 3. Perform convolution calculations via lookups
  // NOTE: this is for a single filter
  {
    constexpr int factor = 1; // changing to 2 didn't make noticeable difference
    constexpr dim3 tpb(32, 16); // 32x8 or 32x16 seems better than 32x4
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    sats::convolve_via_SAT_and_rowSums_naive_kernel<Tin, Tout, Tout, Tout, int,
                                                    int16_t>
        <<<blks, tpb>>>(filter.toDevice(), d_data.cimage(), //
                        preprocessor.d_rowSums().cimage(),  //
                        preprocessor.d_sat().cimage(),      //
                        d_out.image());
  }

  thrust::host_vector<Tout> h_rowSums = preprocessor.d_rowSums().vec;
  thrust::host_vector<Tout> h_sat = preprocessor.d_sat().vec;
  thrust::host_vector<Tout> h_out = d_out.vec;

  // ======================= Check row sums (GOOD)
  printf("====\n");
  printf("Checking initial row sums...\n");
  for (int i = 0; i < height; ++i) {
    int sum = 0;
    for (int j = 0; j < width; ++j) {
      sum += h_data[i * width + j];
      if (sum != h_rowSums[i * width + j]) {
        printf("RowSums Mismatch at (%d, %d): expected %d vs %ld\n", i, j, sum,
               h_rowSums[i * width + j]);
        break;
      }
    }
  }

  // ======================= Check SATs by checking column sums (GOOD)
  printf("Checking SATS by examining column sums...\n");
  for (int j = 0; j < width; ++j) {
    int sum = 0;
    for (int i = 0; i < height; ++i) {
      sum += h_rowSums[i * width + j];
      if (sum != h_sat[i * width + j]) {
        printf("SATS Mismatch at (%d, %d): expected %d vs %ld\n", i, j, sum,
               h_sat[i * width + j]);
        break;
      }
    }
  }

  // ======================= Check outputs
  if (height < 100 && width < 100) {
    printf("Checking outputs (expects all integers)...\n");
    std::vector<double> checkOut = convExplicitly<Tin, double, double>(
        mat, filter.maxDiskLength(), h_data.data(), height, width);

    for (int i = 0; i < (int)checkOut.size(); ++i) {
      if (checkOut[i] != h_out[i]) {
        printf("**** ERROR: Output Mismatch at (%d, %d): expected %f vs %ld\n",
               i / width, i % width, checkOut[i], h_out[i]);
        break;
      }
    }
  }

  if (height <= 32 && width <= 32) {
    printf("-------- Original:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4d ", h_data[i * width + j]);
      }
      printf("\n");
    }

    printf("-------- Row sums:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4ld ", h_rowSums[i * width + j]);
      }
      printf("\n");
    }

    printf("-------- SAT:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4ld ", h_sat[i * width + j]);
      }
      printf("\n");
    }

    printf("-------- Output:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4ld ", h_out[i * width + j]);
      }
      printf("\n");
    }
  }

  // ============================================================
  // ============================================================
  // ============================================================
  // ============================================================
  // ============================================================
  // ============================================================

  // Arbitrary testing for multiple filters
  const int numFilters = 2;
  sats::MultiFilterOfDisksRowSATCreator<int16_t, double> multifilter;
  for (int i = 0; i < numFilters; ++i) {
    double mScaleList[4] = {scaleList[0] + i, scaleList[1] + i,
                            scaleList[2] + i, scaleList[3] + i};
    double mRadiusPixels[4] = {radiusPixels[0] + i, radiusPixels[1] + i,
                               radiusPixels[2] + i, radiusPixels[3] + i};
    multifilter.addFilter(mScaleList, mRadiusPixels, numDisks);
  }
  // Remember to push all the filters up!
  multifilter.h2d();

  // debug prints
  for (size_t i = 0; i < multifilter.numFilters(); ++i) {
    auto &filter = multifilter.h_filters_vec[i];
    printf("Filter %zu has %u disks, at %p\n", i, filter.numDisks,
           filter.d_disks);
  }

  {
    sats::SimpleRule<numFilters> rule;
    constexpr int factor = 1;
    constexpr dim3 tpb(32, 4);
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    sats::convolve_via_SAT_and_rowSums_dynamicFilters_kernel<
        Tin, Tout, Tout, Tout, int, sats::SimpleRule<numFilters>>
        <<<blks, tpb>>>(multifilter.d_filters(), rule, d_data.cimage(), //
                        preprocessor.d_rowSums().cimage(),              //
                        preprocessor.d_sat().cimage(),                  //
                        d_out.image());
  }

  h_out = d_out.vec;

  if (height <= 32 && width <= 32) {
    // print the two filters again
    for (size_t i = 0; i < multifilter.numFilters(); ++i) {
      auto &filter = multifilter.h_filtercreators[i];
      std::vector<double> mat =
          multifilter.h_filtercreators[i].makeMat<double>();

      printf("-------- Filter %zu:\n", i);
      for (int i = 0; i < filter.maxDiskLength(); ++i) {
        for (int j = 0; j < filter.maxDiskLength(); ++j) {
          printf("%.1f ", mat[i * filter.maxDiskLength() + j]);
        }
        printf("\n");
      }
    }

    // Print input again
    printf("-------- Original:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4d ", h_data[i * width + j]);
      }
      printf("\n");
    }

    // Check final output
    printf("-------- Output:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4ld ", h_out[i * width + j]);
      }
      printf("\n");
    }
  }

  return 0;
}

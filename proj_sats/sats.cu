#include <iostream>
#include <random>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
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

  double radiusPixels = 1.1;
  if (argc >= 4)
    radiusPixels = atof(argv[3]);
  printf(" Radius in pixels: %f\n", radiusPixels);

  using Tin = int32_t;  // input is small
  using Tout = int64_t; // use 64-bit for everything else

  thrust::host_vector<Tin> h_data(height * width);
  for (int i = 0; i < height * width; i++) {
    h_data[i] = std::rand() % 10 - 5;
  }

  // thrust::device_vector<Tdata> d_data(h_data);
  containers::DeviceImageStorage<Tin> d_data(width, height);
  thrust::copy(h_data.begin(), h_data.end(), d_data.vec.begin());

  containers::DeviceImageStorage<Tout> d_out(width, height);

  // Pre-compute disk
  // int maxSections =
  //     sats::getMaximumSectionsForDisk_prefixRows_SAT(radiusPixels);
  // thrust::host_vector<sats::DiskSection<int16_t>> h_sections(maxSections);
  // thrust::host_vector<uint8_t> h_sectionTypes(maxSections);
  // int numSections = sats::constructSectionsForDisk_prefixRows_SAT(
  //     radiusPixels, h_sections.data(), h_sectionTypes.data());
  // // Copy disk sections and types to device
  // thrust::device_vector<sats::DiskSection<int16_t>> d_sections(numSections);
  // thrust::device_vector<uint8_t> d_sectionTypes(numSections);
  // thrust::copy(h_sections.begin(), h_sections.begin() + numSections,
  //              d_sections.begin());
  // thrust::copy(h_sectionTypes.begin(), h_sectionTypes.begin() + numSections,
  //              d_sectionTypes.begin());
  // // Make container
  // sats::DiskRowSAT<int16_t> d_disk{1.0, (int)radiusPixels, numSections,
  //                                  d_sections.data().get(),
  //                                  d_sectionTypes.data().get()};

  double scaleList[1] = {1.0};
  double radiusPixelList[1] = {radiusPixels};
  sats::FilterOfDisksRowSATCreator<1, int16_t> filter(scaleList,
                                                      radiusPixelList);

  // Print lots of things to check?
  for (int d = 0; d < filter.getNumDisks(); ++d) {
    auto d_disk = filter.d_disks[d];

    printf("Disk length %d, with %d sections, scale %f\n",
           d_disk.lengthPixels(), d_disk.numSections, d_disk.scale);
    for (int i = 0; i < d_disk.numSections; ++i) {
      printf("Section %d: type %s row %d:%d col %d:%d\n", i,
             sats::sectionTypeString(filter.h_sectionTypes[i]).c_str(),
             filter.h_sections[i].startRow, filter.h_sections[i].endRow,
             -filter.h_sections[i].colOffset, filter.h_sections[i].colOffset);
    }
    printf("-----\n");
    for (int i = -(int)radiusPixels; i < (int)radiusPixels + 1; ++i) {
      auto section = sats::getDiskSectionForRow(filter.h_sections.data().get(),
                                                d_disk.numSections, i);
      printf("Row %d -> col %d : %d\n", i, -section.colOffset,
             section.colOffset);
    }
    printf("-----\n");
  }
  // End of debug printing

  // // Pre-compute multiple disks into a container
  // thrust::host_vector<sats::DiskSection<int16_t>> h_multidiskSections;
  // thrust::host_vector<uint8_t> h_multidiskSectionTypes;
  // thrust::host_vector<int> h_multidiskRadii;
  // int numDisks = 60;
  // // sats::DiskSelectionRule<int16_t> diskRule{0.5f * width,
  // //                                           (int16_t)(0.5f * width),
  // //                                           (int16_t)(0.5f * height),
  // //                                           numDisks};
  // sats::RadialThresholdLinearGradientRule<int16_t> diskRule(
  //     0.5f * width / 150.0f / 3.0f, (int16_t)(0.5f * width),
  //     (int16_t)(0.5f * height), 0.5f * width * 130.0 / 150.0, 0.5f * width);
  // thrust::host_vector<int> h_multidiskNumSections =
  //     sats::constructMultipleDisksViaRule(radiusPixels, h_multidiskSections,
  //                                         h_multidiskSectionTypes,
  //                                         h_multidiskRadii, diskRule);
  //
  // int totalUsedSections = std::accumulate(h_multidiskNumSections.begin(),
  //                                         h_multidiskNumSections.end(), 0);
  // printf("Multidisk sections size used %d / %zu\n", totalUsedSections,
  //        h_multidiskSections.size());
  // printf("Multidisk section types size used %d / %zu\n", totalUsedSections,
  //        h_multidiskSectionTypes.size());
  // // Copy the sections and section types like before
  // thrust::device_vector<sats::DiskSection<int16_t>> d_multidiskSections(
  //     totalUsedSections);
  // thrust::device_vector<uint8_t> d_multidiskSectionTypes(totalUsedSections);
  // thrust::copy(h_multidiskSections.begin(),
  //              h_multidiskSections.begin() + totalUsedSections,
  //              d_multidiskSections.begin());
  // thrust::copy(h_multidiskSectionTypes.begin(),
  //              h_multidiskSectionTypes.begin() + totalUsedSections,
  //              d_multidiskSectionTypes.begin());
  // // Make container
  // thrust::host_vector<sats::DiskRowSAT<int16_t>> h_multidisks(numDisks);
  // std::vector<double> h_diskScales = {1.0, 1.0};
  // sats::createContainer_DiskRowSAT<int16_t>(
  //     h_multidisks.data(), h_diskScales.data(), h_multidiskRadii.data(),
  //     h_multidiskNumSections, d_multidiskSections.data().get(),
  //     d_multidiskSectionTypes.data().get());
  //
  // thrust::device_vector<sats::DiskRowSAT<int16_t>> d_multidisks =
  // h_multidisks;
  //

  sats::DiskConvolver_PrefixRowsSAT<Tin, Tout, Tout> convolver(height, width);
  convolver.preprocess(d_data.vec.data().get());

  // === 3. Perform convolution calculations via lookups
  {
    constexpr int factor = 1; // changing to 2 didn't make noticeable difference
    constexpr dim3 tpb(32, 16); // 32x8 or 32x16 seems better than 32x4
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    sats::convolve_via_SAT_and_rowSums_naive_kernel<1, Tin, Tout, Tout, Tout,
                                                    unsigned int, int16_t>
        <<<blks, tpb>>>(filter.toDevice(), d_data.cimage(), //
                        convolver.d_rowSums().cimage(),     //
                        convolver.d_sat().cimage(),         //
                        d_out.image());
  }

  thrust::host_vector<Tout> h_rowSums = convolver.d_rowSums().vec;
  thrust::host_vector<Tout> h_sat = convolver.d_sat().vec;
  thrust::host_vector<Tout> h_out = d_out.vec;

  // ======================= Check row sums (GOOD)
  printf("====\n");
  printf("Checking initial row sums...\n");
  for (int i = 0; i < height; ++i) {
    int sum = 0;
    for (int j = 0; j < width; ++j) {
      sum += h_data[i * width + j];
      if (sum != h_rowSums[i * width + j]) {
        printf("RowSums Mismatch at (%d, %d): expected %d vs %d\n", i, j, sum,
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
        printf("SATS Mismatch at (%d, %d): expected %d vs %d\n", i, j, sum,
               h_sat[i * width + j]);
        break;
      }
    }
  }

  // ======================= Check outputs
  printf("Checking outputs...\n");
  size_t maxChecks = 10000;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      Tout val = 0;
      for (int r = -(int)radiusPixels; r <= (int)radiusPixels; ++r) {
        // get the section for this offset
        auto section = sats::getDiskSectionForRow(
            filter.h_sections.data().get(), filter.d_disks[0].numSections, r);
        int y = r + i;
        if (y < 0 || y >= height)
          continue;
        for (int x = -section.colOffset + j; x <= section.colOffset + j; ++x) {
          if (x < 0 || x >= width)
            continue;
          // printf("Accessing (%d, %d) for (%d, %d)\n", y, x, i, j);
          val += h_data[y * width + x];
        }
      }
      // assumes scale 1.0
      if (val != h_out[i * width + j]) {
        printf("Output Mismatch at (%d, %d): expected %d vs %d\n", i, j, val,
               h_out[i * width + j]);
        break;
      }
      --maxChecks;
      if (maxChecks == 0) {
        printf("Not checking any more..\n");
        break;
      }
    }
    if (maxChecks == 0) {
      break;
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
        printf("%4d ", h_rowSums[i * width + j]);
      }
      printf("\n");
    }

    printf("-------- SAT:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4d ", h_sat[i * width + j]);
      }
      printf("\n");
    }

    printf("-------- Output:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4d ", h_out[i * width + j]);
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

  // // === 4. Perform convolution calculations via lookups
  // {
  //   diskRule.print();
  //   constexpr int factor = 1;
  //   constexpr dim3 tpb(32, 4);
  //   dim3 blks((width + tpb.x - 1) / tpb.x,
  //             (height + tpb.y - 1) / tpb.y / factor);
  //   convolve_via_SAT_and_rowSums_dynamicDisks_kernel<Tdata, unsigned int,
  //                                                    int16_t, double>
  //       <<<blks, tpb>>>(d_multidisks.data().get(), diskRule, image, rowSums,
  //                       sat, out);
  // }

  return 0;
}

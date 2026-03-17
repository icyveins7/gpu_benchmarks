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

struct IndexToRowFunctor {
  int width = 1;
  __host__ __device__ int operator()(int i) { return i / width; }
};

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

template <typename Tdata, typename Tidx>
__device__ Tdata getSATelement(const containers::Image<Tdata, Tidx> sat, int y,
                               int x) {
  /*
  NOTE: Although the SAT is defined for a given M x N dimensions,
  access outside the bounds should be well-handled, similar to the
  row-prefixes.

  1) Anything with a negative index should be considered to be 0.
    This applies to a, b or c.

  2) For either row/col indices indexing past the ends, the
  end row/col value will be used. This is similar to the prefix sum.

  3) For cases where the row AND col are both out of bounds, the result
  will be the bottom-right value. 'SAT of the whole image + border = SAT
  of the whole image'

  IMPORTANT: you cannot assume that the points a, b, c and d always
  follow some rules because of their direction. This is because the
  rectangle sections themselves are OFFSET from the thread's target
  position. For example, the thread handling the bottom-right most point
  of the image will likely have a rectangle that fully exists below the
  image, with all points a-d being out of bounds to the
  bottom/right/bottomright.

  However, the check is actually pretty simple; regardless of what is out of
  bounds, we need only clip both dimensions to their row/col max index values.
  */

  if (y < 0 || x < 0) {
    return 0;
  }

  // Clip to row and/or col ending values
  return sat.atWithDefaultPointer(
      y, x, &sat.at(min(sat.height - 1, y), min(sat.width - 1, x)));
}

template <typename Tin, typename Trowsum, typename Tsat, typename Tout,
          typename Tidx, typename Tsection = int16_t, typename Tscale = double>
__device__ Tscale sumOverDisk_SAT_and_rowSums_threadwork(
    const sats::DiskRowSAT<Tsection, Tscale> disk,
    const containers::Image<Tin, Tidx> orig,
    const containers::Image<Trowsum, Tidx> rowSums,
    const containers::Image<Tsat, Tidx> sat, const int x, const int y) {
  // Each thread's output value, and then loop over the sections to
  // accumulate
  Tout val = 0;
  for (int i = 0; i < disk.numSectionsToIterate(); ++i) {
    uint8_t sectionType = disk.getSectionType(i);
    sats::DiskSection<Tsection> section = disk.getSection(i);

    // There is no warp divergence here!
    // All threads in a warp will be tackling the same type
    if (sectionType == sats::SectionType::LOOKUP_TYPE_PIXEL) {
      // Simply look up the pixel from the original image
      Tin pixel = orig.atWithDefault(
          y + section.startRow, x, // colOffset should be 0 already
          0);                      // pixels outside the image treated as 0
      val += pixel;

      // // DEBUG
      // if (x == 1 && y == 0) {
      //   printf("%d, %d -> pixel, val + %ld to %ld\n", y + section.startRow,
      //          x + section.startCol, pixel, val);
      // }
    } else if (sectionType == sats::SectionType::LOOKUP_TYPE_ROW) {
      // Look up the row, and then the columns
      int row = y + section.startRow;       // same as end row by definition
      int left = x - section.colOffset - 1; // exclude starting col
      int right = x + section.colOffset;

      if (rowSums.rowIsValid(row)) {
        // A valid row should treat the negative indices as 0 for prefix
        // sums and it should treat the out of range on the right as the
        // final col value
        // Trowsum a = rowSums.atWithDefault(row, left, 0);
        // Trowsum b =
        //     rowSums.atWithDefaultPointer(row, right, &rowSums.rowEnd(row));

        // This profiles slightly faster, probably because less redundant checks
        Trowsum a = 0; // default
        if (rowSums.colIsValid(left))
          a = rowSums.at(row, left);
        Trowsum b;
        if (rowSums.colIsValid(right))
          b = rowSums.at(row, right);
        else
          b = rowSums.rowEnd(row);

        val += b - a;
        // // DEBUG
        // if (x == 1 && y == 0) {
        //   printf("row %d, col %d : %d -> row, val + %ld to %ld\n", row, left,
        //          right, b - a, val);
        // }
      }

    } else if (sectionType == sats::SectionType::LOOKUP_TYPE_RECT) {
      // Look up the SAT, assuming a y descending format:
      //       a -------- b
      //       |          |
      //       |          |
      //       c -------- d
      int2 topLeft =
          make_int2(x - section.colOffset - 1, y + section.startRow - 1);
      Tsat a = getSATelement(sat, topLeft.y, topLeft.x);

      int2 topRight =
          make_int2(x + section.colOffset, y + section.startRow - 1);
      Tsat b = getSATelement(sat, topRight.y, topRight.x);

      int2 bottomLeft =
          make_int2(x - section.colOffset - 1, y + section.endRow);
      Tsat c = getSATelement(sat, bottomLeft.y, bottomLeft.x);

      int2 bottomRight = make_int2(x + section.colOffset, y + section.endRow);
      Tsat d = getSATelement(sat, bottomRight.y, bottomRight.x);

      val += a + d - b - c;
    }
  } // end loop over disk sections

  return val;
}

template <typename Tin, typename Trowsum, typename Tsat, typename Tout,
          typename Tidx, typename Tsection = int16_t, typename Tscale = double,
          bool incrementInsteadOfSet = false>
__global__ void convolve_via_SAT_and_rowSums_naive_kernel(
    const sats::DiskRowSAT<Tsection, Tscale> disk,
    const containers::Image<Tin, Tidx> orig,
    const containers::Image<Trowsum, Tidx> rowSums,
    const containers::Image<Tsat, Tidx> sat,
    containers::Image<Tout, Tidx> out) {

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < (int)orig.height;
       y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < (int)orig.width;
         x += blockDim.x * gridDim.x) {

      Tscale val =
          sumOverDisk_SAT_and_rowSums_threadwork<Tin, Trowsum, Tsat, Tout, Tidx,
                                                 Tsection, Tscale>(
              disk, orig, rowSums, sat, x, y);

      // Value is ready here, write it back
      if constexpr (incrementInsteadOfSet)
        out.at(y, x) += static_cast<Tout>(val * disk.scale);
      else
        out.at(y, x) = static_cast<Tout>(val * disk.scale);
    }
  }
}

template <typename Tdata, typename Tidx, typename Tsection = int16_t,
          typename Tscale = double, typename Trule>
__global__ void convolve_via_SAT_and_rowSums_dynamicDisks_kernel(
    const sats::DiskRowSAT<Tsection, Tscale> *disks, const Trule rule,
    const containers::Image<Tdata, Tidx> orig,
    const containers::Image<Tdata, Tidx> rowSums,
    const containers::Image<Tdata, Tidx> sat,
    containers::Image<Tdata, Tidx> out) {

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < orig.height;
       y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < orig.width;
         x += blockDim.x * gridDim.x) {

      // Determine the disk required
      // Example here is to use radius-based linear steps starting from the
      // centre to determine the disk index. Any other rule should work
      unsigned int diskIndex = rule.getDiskIndex(y, x);
      const sats::DiskRowSAT<Tsection, Tscale> &disk = disks[diskIndex];

      // Every thread (potentially) uses its own disk
      Tscale val =
          sumOverDisk_SAT_and_rowSums_threadwork<Tdata, Tidx, Tsection, Tscale>(
              disk, orig, rowSums, sat, x, y);

      // if (threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf(" Disk index for (row %d, col %d) is %d -> %d sections\n", y,
      //   x,
      //          diskIndex, disk.numSections);
      // }

      // Value is ready here, write it back
      out.at(y, x) = static_cast<Tdata>(val * disk.scale);
    }
  }
}

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
  thrust::host_vector<Tout> h_rowSums(height * width);
  thrust::host_vector<Tout> h_sat(height * width);
  thrust::host_vector<Tout> h_out(height * width);

  // thrust::device_vector<Tdata> d_data(h_data);
  containers::DeviceImageStorage<Tin> d_data(width, height);
  thrust::copy(h_data.begin(), h_data.end(), d_data.vec.begin());

  containers::DeviceImageStorage<Tout> d_rowSums(width, height);
  containers::DeviceImageStorage<Tout> d_transpose(width, height);
  containers::DeviceImageStorage<Tout> d_transpose2(width, height);
  containers::DeviceImageStorage<Tout> d_sat(width, height);
  containers::DeviceImageStorage<Tout> d_out(width, height);

  // CUB related prep
  auto rowKeyIterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{width});
  cubw::DeviceScan::InclusiveSumByKey<decltype(rowKeyIterator), Tin *, Tout *>
      cubwRowScan(width * height);

  auto colTransposeKeyIterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{height});
  cubw::DeviceScan::InclusiveSumByKey<decltype(colTransposeKeyIterator), Tout *,
                                      Tout *>
      cubwColScanTranspose(height * width);

  // Pre-compute disk
  int maxSections =
      sats::getMaximumSectionsForDisk_prefixRows_SAT(radiusPixels);
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections(maxSections);
  thrust::host_vector<uint8_t> h_sectionTypes(maxSections);
  int numSections = sats::constructSectionsForDisk_prefixRows_SAT(
      radiusPixels, h_sections.data(), h_sectionTypes.data());
  // Copy disk sections and types to device
  thrust::device_vector<sats::DiskSection<int16_t>> d_sections(numSections);
  thrust::device_vector<uint8_t> d_sectionTypes(numSections);
  thrust::copy(h_sections.begin(), h_sections.begin() + numSections,
               d_sections.begin());
  thrust::copy(h_sectionTypes.begin(), h_sectionTypes.begin() + numSections,
               d_sectionTypes.begin());
  // Make container
  sats::DiskRowSAT<int16_t> d_disk{1.0, (int)radiusPixels, numSections,
                                   d_sections.data().get(),
                                   d_sectionTypes.data().get()};
  printf("Disk length %d, with %d sections, scale %f\n", d_disk.lengthPixels(),
         d_disk.numSections, d_disk.scale);
  for (int i = 0; i < numSections; ++i) {
    printf("Section %d: type %s row %d:%d col %d:%d\n", i,
           sats::sectionTypeString(h_sectionTypes[i]).c_str(),
           h_sections[i].startRow, h_sections[i].endRow,
           -h_sections[i].colOffset, h_sections[i].colOffset);
  }
  printf("-----\n");
  for (int i = -(int)radiusPixels; i < (int)radiusPixels + 1; ++i) {
    auto section =
        sats::getDiskSectionForRow(h_sections.data(), numSections, i);
    printf("Row %d -> col %d : %d\n", i, -section.colOffset, section.colOffset);
  }
  printf("-----\n");

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
  // === 1. Perform prefix sums across rows

  // Custom kernel method (not enough occupancy to do this, must optimize
  // beyond 1 block per row)
  // {
  //   constexpr int tpb = 256;
  //   int blks = (height + tpb - 1) / tpb;
  //   computeSATkernel<int, int, tpb><<<blks, tpb>>>(image, rowSums, sat);
  // }

  // Generic CUB routine
  cubwRowScan.exec(rowKeyIterator, d_data.vec.data().get(),
                   d_rowSums.vec.data().get(), width * height);

  // === 2a. Perform prefix sums across columns (via explicitly transposed
  // matrix) Transpose into the SAT matrix
  transpose<Tout>(d_transpose.vec.data().get(), d_rowSums.vec.data().get(),
                  (int)height, (int)width);
  // In-place row-sums on the transpose (width is now the original height)
  cubwColScanTranspose.exec(colTransposeKeyIterator,
                            d_transpose.vec.data().get(),
                            d_transpose2.vec.data().get(), width * height);
  // Transposing back
  transpose<Tout>(d_sat.vec.data().get(), d_transpose2.vec.data().get(), width,
                  height);

  // === 3. Perform convolution calculations via lookups
  {
    constexpr int factor = 1; // changing to 2 didn't make noticeable difference
    constexpr dim3 tpb(32, 16); // 32x8 or 32x16 seems better than 32x4
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    convolve_via_SAT_and_rowSums_naive_kernel<Tin, Tout, Tout, Tout,
                                              unsigned int, int16_t>
        <<<blks, tpb>>>(d_disk, d_data.image(), d_rowSums.image(),
                        d_sat.image(), d_out.image());
  }

  thrust::host_vector<int> h_transpose = d_transpose.vec;
  thrust::host_vector<int> h_transpose2 = d_transpose2.vec;
  h_rowSums = d_rowSums.vec;
  h_sat = d_sat.vec;
  h_out = d_out.vec;

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

  // ======================= Check row sums across transposed matrix (GOOD)
  printf("Checking row sums across initial row sums transposed...\n");
  for (int i = 0; i < width; ++i) { // there are 'width' rows now
    int sum = 0;
    for (int j = 0; j < height; ++j) { // and 'height' columns
      sum += h_transpose[i * height + j];
      if (sum != h_transpose2[i * height + j]) {
        printf("RowSums On TransposedRowSums Mismatch at (%d, %d): expected %d "
               "vs %d\n",
               i, j, sum, h_transpose2[i * height + j]);
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
        auto section =
            sats::getDiskSectionForRow(h_sections.data(), numSections, r);
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

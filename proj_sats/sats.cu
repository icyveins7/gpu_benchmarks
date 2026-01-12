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

template <typename Tdata, typename Tidx, int BLOCK_THREADS>
__device__ void computeRowPrefixSums_BlockPerRow(
    containers::Image<Tdata, Tidx> image,
    containers::Image<Tdata, Tidx> rowSums,
    typename cub::BlockScan<Tdata, BLOCK_THREADS>::TempStorage &temp_storage) {
  using BlockScan = cub::BlockScan<Tdata, BLOCK_THREADS>;

  // Each block does one row
  for (int i = blockIdx.x; i < image.height; i += gridDim.x) {
    // Define a carryover for this row
    Tdata carryover = 0;

    // Read elements in the row
    for (int j = 0; j < image.width; j += blockDim.x) {
      Tdata tElem = 0;           // default value
      Tidx jt = j + threadIdx.x; // column for this thread
      if (image.colIsValid(jt)) {
        tElem = image.at(i, jt);
      }
      // CUB is great, no need to syncthreads since it performs internal warp
      // syncs
      Tdata newCarryover;
      BlockScan(temp_storage)
          .InclusiveSum(tElem, tElem, newCarryover); // in-place write

      // Add carryover
      tElem += carryover;

      // Also add the new carryover
      carryover += newCarryover;

      // Write back
      if (image.colIsValid(jt)) {
        rowSums.at(i, jt) = tElem;
      }

      // In order to reuse everything, we need to sync
      __syncthreads();
    }
  }
}

template <typename Tdata, typename Tidx, int BLOCK_THREADS>
__global__ void computeSATkernel(containers::Image<Tdata, Tidx> image,
                                 containers::Image<Tdata, Tidx> rowSums,
                                 containers::Image<Tdata, Tidx> sat) {
  using BlockScan = cub::BlockScan<Tdata, BLOCK_THREADS>;
  using TempStorage = typename BlockScan::TempStorage;

  __shared__ TempStorage temp_storage;

  computeRowPrefixSums_BlockPerRow<Tdata, Tidx, BLOCK_THREADS>(image, rowSums,
                                                               temp_storage);

  // TODO: do the columns
}

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

template <typename Tdata, typename Tidx, typename Tsection = int16_t,
          typename Tscale = double>
__device__ Tscale sumOverDisk_SAT_and_rowSums_threadwork(
    const sats::DiskRowSAT<Tsection, Tscale> disk,
    const containers::Image<Tdata, Tidx> orig,
    const containers::Image<Tdata, Tidx> rowSums,
    const containers::Image<Tdata, Tidx> sat, const int x, const int y) {
  // Each thread's output value, and then loop over the sections to
  // accumulate
  Tdata val = 0;
  for (int i = 0; i < disk.numSectionsToIterate(); ++i) {
    uint8_t sectionType = disk.getSectionType(i);
    sats::DiskSection<Tsection> section = disk.getSection(i);

    // There is no warp divergence here!
    // All threads in a warp will be tackling the same type
    if (sectionType == LOOKUP_TYPE_PIXEL) {
      // Simply look up the pixel from the original image
      val += orig.atWithDefault(y + section.startRow, x + section.startCol,
                                0); // pixels outside the image treated as 0
    } else if (sectionType == LOOKUP_TYPE_ROW) {
      // Look up the row, and then the columns
      int row = y + section.startRow;
      int left = x + section.startCol - 1; // exclude starting col
      int right = x + section.endCol;

      if (rowSums.rowIsValid(row)) {
        // A valid row should treat the negative indices as 0 for prefix
        // sums and it should treat the out of range on the right as the
        // final col value
        Tdata a = rowSums.atWithDefault(row, left, 0);
        Tdata b =
            rowSums.atWithDefaultPointer(row, right, &rowSums.rowEnd(row));
        val += b - a;
      }

    } else if (sectionType == LOOKUP_TYPE_RECT) {
      // Look up the SAT, assuming a y descending format:
      //       a -------- b
      //       |          |
      //       |          |
      //       c -------- d
      int2 topLeft =
          make_int2(x + section.startCol - 1, y + section.startRow - 1);
      Tdata a = getSATelement(sat, topLeft.y, topLeft.x);

      int2 topRight = make_int2(x + section.endCol, y + section.startRow - 1);
      Tdata b = getSATelement(sat, topRight.y, topRight.x);

      int2 bottomLeft = make_int2(x + section.startCol - 1, y + section.endRow);
      Tdata c = getSATelement(sat, bottomLeft.y, bottomLeft.x);

      int2 bottomRight = make_int2(x + section.endCol, y + section.endRow);
      Tdata d = getSATelement(sat, bottomRight.y, bottomRight.x);

      val += a + d - b - c;
    }
  } // end loop over disk sections

  return val;
}

template <typename Tdata, typename Tidx, typename Tsection = int16_t,
          typename Tscale = double>
__global__ void convolve_via_SAT_and_rowSums_naive_kernel(
    const sats::DiskRowSAT<Tsection, Tscale> disk,
    const containers::Image<Tdata, Tidx> orig,
    const containers::Image<Tdata, Tidx> rowSums,
    const containers::Image<Tdata, Tidx> sat,
    containers::Image<Tdata, Tidx> out) {

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < (int)orig.height;
       y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < (int)orig.width;
         x += blockDim.x * gridDim.x) {

      Tscale val =
          sumOverDisk_SAT_and_rowSums_threadwork<Tdata, Tidx, Tsection, Tscale>(
              disk, orig, rowSums, sat, x, y);

      // Value is ready here, write it back
      out.at(y, x) = static_cast<Tdata>(val * disk.scale);
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

  int height = 16, width = 16;
  if (argc >= 3) {
    height = atoi(argv[1]);
    width = atoi(argv[2]);
  }
  printf(" Image size (rows x columns): %d x %d\n", height, width);

  int radiusPixels = 3;
  if (argc >= 4)
    radiusPixels = atoi(argv[3]);
  printf(" Radius in pixels: %d\n", radiusPixels);

  thrust::host_vector<int> h_data(height * width);
  for (int i = 0; i < height * width; i++) {
    h_data[i] = std::rand() % 10;
  }
  thrust::host_vector<int> h_rowSums(height * width);
  thrust::host_vector<int> h_sat(height * width);
  thrust::host_vector<int> h_out(height * width);

  thrust::device_vector<int> d_data(h_data);
  thrust::device_vector<int> d_rowSums(h_data.size());
  thrust::device_vector<int> d_transpose(h_data.size());
  thrust::device_vector<int> d_transpose2(h_data.size());
  thrust::device_vector<int> d_sat(h_data.size());
  thrust::device_vector<int> d_out(h_data.size());

  containers::Image<int, unsigned int> image(d_data.data().get(), width,
                                             height);
  containers::Image<int, unsigned int> rowSums(d_rowSums.data().get(), width,
                                               height);
  containers::Image<int, unsigned int> transposeImage(d_transpose.data().get(),
                                                      height, width);
  containers::Image<int, unsigned int> sat(d_sat.data().get(), width, height);
  containers::Image<int, unsigned int> out(d_out.data().get(), width, height);

  // CUB related prep
  auto rowKeyIterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{width});
  cubw::DeviceScan::InclusiveSumByKey<decltype(rowKeyIterator), int *, int *>
      cubwRowScan(width * height);

  auto colTransposeKeyIterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{height});
  cubw::DeviceScan::InclusiveSumByKey<decltype(colTransposeKeyIterator), int *,
                                      int *>
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
  sats::DiskRowSAT<int16_t> d_disk{1.0, radiusPixels, numSections,
                                   d_sections.data().get(),
                                   d_sectionTypes.data().get()};
  printf("Disk length %d, with %d sections\n", d_disk.lengthPixels(),
         d_disk.numSections);

  // Pre-compute multiple disks into a container
  thrust::host_vector<sats::DiskSection<int16_t>> h_multidiskSections;
  thrust::host_vector<uint8_t> h_multidiskSectionTypes;
  thrust::host_vector<int> h_multidiskRadii;
  int numDisks = 60;
  // sats::DiskSelectionRule<int16_t> diskRule{0.5f * width,
  //                                           (int16_t)(0.5f * width),
  //                                           (int16_t)(0.5f * height),
  //                                           numDisks};
  sats::RadialThresholdLinearGradientRule<int16_t> diskRule(
      0.5f * width / 150.0f / 3.0f, (int16_t)(0.5f * width),
      (int16_t)(0.5f * height), 0.5f * width * 130.0 / 150.0, 0.5f * width);
  thrust::host_vector<int> h_multidiskNumSections =
      sats::constructMultipleDisksViaRule(radiusPixels, h_multidiskSections,
                                          h_multidiskSectionTypes,
                                          h_multidiskRadii, diskRule);

  int totalUsedSections = std::accumulate(h_multidiskNumSections.begin(),
                                          h_multidiskNumSections.end(), 0);
  printf("Multidisk sections size used %d / %zu\n", totalUsedSections,
         h_multidiskSections.size());
  printf("Multidisk section types size used %d / %zu\n", totalUsedSections,
         h_multidiskSectionTypes.size());
  // Copy the sections and section types like before
  thrust::device_vector<sats::DiskSection<int16_t>> d_multidiskSections(
      totalUsedSections);
  thrust::device_vector<uint8_t> d_multidiskSectionTypes(totalUsedSections);
  thrust::copy(h_multidiskSections.begin(),
               h_multidiskSections.begin() + totalUsedSections,
               d_multidiskSections.begin());
  thrust::copy(h_multidiskSectionTypes.begin(),
               h_multidiskSectionTypes.begin() + totalUsedSections,
               d_multidiskSectionTypes.begin());
  // Make container
  thrust::host_vector<sats::DiskRowSAT<int16_t>> h_multidisks(numDisks);
  std::vector<double> h_diskScales = {1.0, 1.0};
  sats::createContainer_DiskRowSAT<int16_t>(
      h_multidisks.data(), h_diskScales.data(), h_multidiskRadii.data(),
      h_multidiskNumSections, d_multidiskSections.data().get(),
      d_multidiskSectionTypes.data().get());

  thrust::device_vector<sats::DiskRowSAT<int16_t>> d_multidisks = h_multidisks;

  // === 1. Perform prefix sums across rows

  // Custom kernel method (not enough occupancy to do this, must optimize
  // beyond 1 block per row)
  // {
  //   constexpr int tpb = 256;
  //   int blks = (height + tpb - 1) / tpb;
  //   computeSATkernel<int, int, tpb><<<blks, tpb>>>(image, rowSums, sat);
  // }

  // Generic CUB routine
  cubwRowScan.exec(rowKeyIterator, d_data.data().get(), d_rowSums.data().get(),
                   width * height);

  // === 2a. Perform prefix sums across columns (via explicitly transposed
  // matrix) Transpose into the SAT matrix
  transpose<int>(d_transpose.data().get(), d_rowSums.data().get(), (int)height,
                 (int)width);
  // In-place row-sums on the transpose (width is now the original height)
  cubwColScanTranspose.exec(colTransposeKeyIterator, d_transpose.data().get(),
                            d_transpose2.data().get(), width * height);
  // Transposing back
  transpose<int>(d_sat.data().get(), d_transpose2.data().get(), width, height);

  // === 3. Perform convolution calculations via lookups
  {
    constexpr int factor = 1;
    constexpr dim3 tpb(32, 4);
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    convolve_via_SAT_and_rowSums_naive_kernel<int, unsigned int, int16_t>
        <<<blks, tpb>>>(d_disk, image, rowSums, sat, out);
  }

  thrust::host_vector<int> h_transpose = d_transpose;
  thrust::host_vector<int> h_transpose2 = d_transpose2;
  h_rowSums = d_rowSums;
  h_sat = d_sat;
  h_out = d_out;

  // ======================= Check row sums (GOOD)
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
  }

  // ============================================================
  // ============================================================
  // ============================================================
  // ============================================================
  // ============================================================
  // ============================================================

  // === 4. Perform convolution calculations via lookups
  {
    diskRule.print();
    constexpr int factor = 1;
    constexpr dim3 tpb(32, 4);
    dim3 blks((width + tpb.x - 1) / tpb.x,
              (height + tpb.y - 1) / tpb.y / factor);
    convolve_via_SAT_and_rowSums_dynamicDisks_kernel<int, unsigned int, int16_t,
                                                     double><<<blks, tpb>>>(
        d_multidisks.data().get(), diskRule, image, rowSums, sat, out);
  }

  return 0;
}

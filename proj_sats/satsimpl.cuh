#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include <pinnedalloc.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
#include "transpose.cuh"

#define LOOKUP_TYPE_PIXEL 0
#define LOOKUP_TYPE_ROW 0
#define LOOKUP_TYPE_COL 0
#define LOOKUP_TYPE_RECT 0

namespace sats {

template <typename T = int16_t> struct DiskSection {
  static_assert(std::is_integral_v<T>,
                "T must be an integer type for ConstantDiskSection");
  static_assert(std::is_signed_v<T>,
                "T must be signed for ConstantDiskSection");

  T startRow;
  T endRow; // inclusive
  T startCol;
  T endCol; // inclusive

  __host__ __device__ T widthPixels() const { return endCol - startCol + 1; }
  __host__ __device__ T heightPixels() const { return endRow - startRow + 1; }

  __host__ __device__ uint8_t type() const {
    if (startRow == endRow) {
      if (startCol == endCol) {
        return LOOKUP_TYPE_PIXEL;
      } else {
        return LOOKUP_TYPE_ROW;
      }
    } else {
      if (startCol == endCol) {
        return LOOKUP_TYPE_COL;
      } else {
        return LOOKUP_TYPE_RECT;
      }
    }
  }
};

/**
 * @brief Disk that can be operated using prefix sums and SAT alone.
 *
 * @detail The sections are ordered in an incrementing row-order. The only types
 * that should be present are PIXEL, ROW and RECT. Sections are only specified
 * up to the middle section (which is assumed to be included, and is assumed to
 * be non-repeated, and should be a ROW).
 *
 * @example
 * For radiusPixels of 3, there are 3 sections specified,
 * even though there are 5 sections in total
 *
 * X X X O X X X --> PIXEL
 * X O O O O O X -->
 * X O O O O O X --> RECT
 * O O O O O O O --> ROW
 * X O O O O O X -->
 * X O O O O O X --> RECT  (Mirrored, not included in data)
 * X X X O X X X --> PIXEL (Mirrored, not included in data)
 *
 * @tparam T Internal type for DiskSections
 */
template <typename T, typename Tscale = double> struct DiskRowSAT {
  Tscale scale = 1.0;
  int radiusPixels; // TODO: maybe i don't need this?
  int numSections;
  const DiskSection<T>
      *sections; // there should be at most (radiusPixels + 1) sections
  const uint8_t *sectionTypes; // matched with sections

  __host__ __device__ int lengthPixels() const { return radiusPixels * 2 + 1; }
  __host__ __device__ static int lengthPixels(int radiusPixels) {
    return radiusPixels * 2 + 1;
  }
  __host__ __device__ int numSectionsToIterate() const {
    return numSections * 2 - 1;
  }
  __host__ __device__ bool validSectionIndex(int i) const {
    return i >= 0 && i < numSectionsToIterate();
  }

  /**
   * @brief Retrieves access index assuming a mirror around the final specified
   section.
   * @detail Assumes input index has been validated (see validSectionIndex()).
   * @example
   * Example: numSections = 3
   * 0 1 2 3 4
   *     |
   *     numSections up to here
   *
   * We require
   * accessIndex(3) -> 1
   * accessIndex(4) -> 0
   *
   * Formula is:
   * numSections - 1 - (i - numSections + 1) =
   * 2 * numSections - 2 - i
   *
   * @param i Input index
   * @return Appropriate access index for the section
   */
  __host__ __device__ int accessIndex(int i) const {
    if (i >= numSections) {
      return 2 * numSections - 2 - i;
    } else {
      return i;
    }
  }

  /**
   * @brief Retrieves section assuming a mirror around the final specified
   * section.
   * @detail Assumes input index has been validated (see validSectionIndex()).
   *
   * @param i Input index
   * @return Appropriate section
   */
  __host__ __device__ const DiskSection<T> &getSection(int i) const {
    return sections[accessIndex(i)];
  }

  /**
   * @brief Retrieves section type assuming a mirror around the final specified
   * section.
   * @detail Assumes input index has been validated (see validSectionIndex()).
   *
   * @param i Input index
   * @return Appropriate section type
   */
  __host__ __device__ uint8_t getSectionType(int i) const {
    return sectionTypes[accessIndex(i)];
  }
};

int getMaximumSectionsForDisk_prefixRows_SAT(const int radiusPixels) {
  return radiusPixels + 1;
}

template <typename T>
int constructSectionsForDisk_prefixRows_SAT(const int radiusPixels,
                                            DiskSection<T> *sections,
                                            uint8_t *sectionTypes) {
  int numSections = 0;

  std::vector<T> rowOffsets(radiusPixels + 1);

  // Go down the rows, including the middle row
  for (size_t i = 0; i < rowOffsets.size(); ++i) {
    double y =
        -radiusPixels + (double)i; // include offset i.e. assume centre is 0,0
    double x = std::sqrt(radiusPixels * radiusPixels - y * y); // right root
    rowOffsets.at(i) = (T)x; // round towards zero
  }

  // Now combine rows that have the same length i.e. offset
  DiskSection<T> section{0, 0, (T)(-rowOffsets.at(0)), rowOffsets.at(0)};
  for (size_t i = 1; i < rowOffsets.size(); ++i) {
    if (rowOffsets.at(i) == rowOffsets.at(i - 1)) {
      // Change the current section
      section.endRow = i;
    } else {
      // Push current section
      sections[numSections] = section;
      sectionTypes[numSections] = section.type();
      numSections++;
      // Make a new section
      section = {(T)i, (T)i, (T)(-rowOffsets.at(i)), rowOffsets.at(i)};
    }
  }
  // Push last section
  sections[numSections] = section;
  sectionTypes[numSections] = section.type();
  numSections++;

  return numSections;
}

template <typename Tidx> struct DiskSelectionRule {
  float stepSize;
  Tidx centreX;
  Tidx centreY;
  unsigned int maxDisks;

  DiskSelectionRule(float _stepSize, Tidx _centreX, Tidx _centreY,
                    int _maxDisks) {
    stepSize = _stepSize;
    centreX = _centreX;
    centreY = _centreY;
    maxDisks = _maxDisks;
  }

  __host__ __device__ unsigned int getDiskIndex(int y, int x) const {
    float dy = y - centreY;
    float dx = x - centreX;
    float radius = sqrtf(dy * dy + dx * dx);
    return min(maxDisks - 1, (int)(radius / stepSize));
  }

  __host__ int getLinearlyShrinkingDiskRadius(int index,
                                              int radiusPixels) const {
    return (double)((maxDisks - index) * radiusPixels) / (double)maxDisks;
  }

  __host__ void print() const {
    printf("Disk Selection Rule:\n"
           "   stepSize: %f\n"
           "   centreX: %d\n"
           "   centreY: %d\n"
           "   maxDisks: %d\n",
           stepSize, centreX, centreY, maxDisks);
  }
};

template <typename Tidx>
struct RadialThresholdLinearGradientRule : public DiskSelectionRule<Tidx> {
  float minRadius; // disk index is 0 until this radius
  float maxRadius; // last disk index is hit at this radius

  RadialThresholdLinearGradientRule(float _stepSize, Tidx _centreX,
                                    Tidx _centreY, float _minRadius,
                                    float _maxRadius)
      : DiskSelectionRule<Tidx>(
            _stepSize, _centreX, _centreY,
            (unsigned int)((_maxRadius - _minRadius) / (_stepSize) + 1)) {
    minRadius = _minRadius;
    maxRadius = _maxRadius;
  }

  __host__ __device__ unsigned int getDiskIndex(int y, int x) const {
    float dy = y - this->centreY;
    float dx = x - this->centreX;
    float radius = sqrtf(dy * dy + dx * dx);

    if (radius < minRadius) {
      return 0;
    } else {
      float incRadius = radius - minRadius;
      return min(this->maxDisks - 1,
                 (unsigned int)(incRadius / this->stepSize));
    }
  }
};

/**
 * @brief This function is used only as a test/example to show how to generate
 * multiple disks from a rule. It should be used as a guide, not as a
 * generalizable method.
 */
template <typename T>
thrust::host_vector<int> constructMultipleDisksViaRule(
    const int radiusPixels, thrust::host_vector<DiskSection<T>> &h_sections,
    thrust::host_vector<uint8_t> &h_sectionTypes,
    thrust::host_vector<int> &h_diskRadii, const DiskSelectionRule<T> rule) {

  std::vector<int> maxNumSections(rule.maxDisks);
  h_diskRadii.resize(rule.maxDisks);

  // Create disk radii
  int maxNumSectionsTotal = 0;
  for (int i = 0; i < (int)rule.maxDisks; ++i) {
    h_diskRadii[i] = rule.getLinearlyShrinkingDiskRadius(i, radiusPixels) + 1;
    maxNumSections.at(i) =
        getMaximumSectionsForDisk_prefixRows_SAT(h_diskRadii[i]);

    printf("Disk %d: radius %d, %d max sections\n", i, h_diskRadii[i],
           maxNumSections.at(i));

    maxNumSectionsTotal += maxNumSections.at(i);
  }

  h_sections.resize(maxNumSectionsTotal);
  h_sectionTypes.resize(maxNumSectionsTotal);
  thrust::host_vector<int> h_numSectionsPerDisk(rule.maxDisks);

  // Now actually create the sections
  int numSectionTotal = 0;
  for (int i = 0; i < (int)rule.maxDisks; ++i) {

    int numSections = constructSectionsForDisk_prefixRows_SAT(
        h_diskRadii[i], &h_sections[numSectionTotal],
        &h_sectionTypes[numSectionTotal]);

    printf("Disk %d: actual sections %d\n", i, numSections);

    h_numSectionsPerDisk[i] = numSections;

    numSectionTotal += numSections;
  }

  return h_numSectionsPerDisk;
}

template <typename T>
void createContainer_DiskRowSAT(
    DiskRowSAT<T> *disks, const double *diskScales, const int *diskRadiusPixels,
    const thrust::host_vector<int> &numSectionsPerDisk,
    DiskSection<T> *d_sections, const uint8_t *d_sectionTypes) {
  int totalSectionsSoFar = 0;
  for (int i = 0; i < (int)numSectionsPerDisk.size(); ++i) {
    int numSections = numSectionsPerDisk[i];
    disks[i] = {diskScales[i], diskRadiusPixels[i], numSectionsPerDisk[i],
                d_sections + totalSectionsSoFar,
                d_sectionTypes + totalSectionsSoFar};
    totalSectionsSoFar += numSections;
  }
}

// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================
// ===========================================================================

struct IndexToRowFunctor {
  int width = 1;
  __host__ __device__ int operator()(int i) { return i / width; }
};

// template <typename Tdata, typename Tidx>
// __device__ Tdata getSATelement(const containers::Image<Tdata, Tidx> sat, int
// y,
//                                int x) {
//   /*
//   NOTE: Although the SAT is defined for a given M x N dimensions,
//   access outside the bounds should be well-handled, similar to the
//   row-prefixes.
//
//   1) Anything with a negative index should be considered to be 0.
//     This applies to a, b or c.
//
//   2) For either row/col indices indexing past the ends, the
//   end row/col value will be used. This is similar to the prefix sum.
//
//   3) For cases where the row AND col are both out of bounds, the result
//   will be the bottom-right value. 'SAT of the whole image + border = SAT
//   of the whole image'
//
//   IMPORTANT: you cannot assume that the points a, b, c and d always
//   follow some rules because of their direction. This is because the
//   rectangle sections themselves are OFFSET from the thread's target
//   position. For example, the thread handling the bottom-right most point
//   of the image will likely have a rectangle that fully exists below the
//   image, with all points a-d being out of bounds to the
//   bottom/right/bottomright.
//
//   However, the check is actually pretty simple; regardless of what is out of
//   bounds, we need only clip both dimensions to their row/col max index
//   values.
//   */
//
//   if (y < 0 || x < 0) {
//     return 0;
//   }
//
//   // Clip to row and/or col ending values
//   return sat.atWithDefaultPointer(
//       y, x, &sat.at(min(sat.height - 1, y), min(sat.width - 1, x)));
// }
//
// template <typename Tdata, typename Tidx, typename Tsection = int16_t,
//           typename Tscale = double>
// __device__ Tscale sumOverDisk_SAT_and_rowSums_threadwork(
//     const sats::DiskRowSAT<Tsection, Tscale> disk,
//     const containers::Image<Tdata, Tidx> orig,
//     const containers::Image<Tdata, Tidx> rowSums,
//     const containers::Image<Tdata, Tidx> sat, const int x, const int y) {
//   // Each thread's output value, and then loop over the sections to
//   // accumulate
//   Tdata val = 0;
//   for (int i = 0; i < disk.numSectionsToIterate(); ++i) {
//     uint8_t sectionType = disk.getSectionType(i);
//     sats::DiskSection<Tsection> section = disk.getSection(i);
//
//     // There is no warp divergence here!
//     // All threads in a warp will be tackling the same type
//     if (sectionType == LOOKUP_TYPE_PIXEL) {
//       // Simply look up the pixel from the original image
//       val += orig.atWithDefault(y + section.startRow, x + section.startCol,
//                                 0); // pixels outside the image treated as 0
//     } else if (sectionType == LOOKUP_TYPE_ROW) {
//       // Look up the row, and then the columns
//       int row = y + section.startRow;
//       int left = x + section.startCol - 1; // exclude starting col
//       int right = x + section.endCol;
//
//       if (rowSums.rowIsValid(row)) {
//         // A valid row should treat the negative indices as 0 for prefix
//         // sums and it should treat the out of range on the right as the
//         // final col value
//         Tdata a = rowSums.atWithDefault(row, left, 0);
//         Tdata b =
//             rowSums.atWithDefaultPointer(row, right, &rowSums.rowEnd(row));
//         val += b - a;
//       }
//
//     } else if (sectionType == LOOKUP_TYPE_RECT) {
//       // Look up the SAT, assuming a y descending format:
//       //       a -------- b
//       //       |          |
//       //       |          |
//       //       c -------- d
//       int2 topLeft =
//           make_int2(x + section.startCol - 1, y + section.startRow - 1);
//       Tdata a = getSATelement(sat, topLeft.y, topLeft.x);
//
//       int2 topRight = make_int2(x + section.endCol, y + section.startRow -
//       1); Tdata b = getSATelement(sat, topRight.y, topRight.x);
//
//       int2 bottomLeft = make_int2(x + section.startCol - 1, y +
//       section.endRow); Tdata c = getSATelement(sat, bottomLeft.y,
//       bottomLeft.x);
//
//       int2 bottomRight = make_int2(x + section.endCol, y + section.endRow);
//       Tdata d = getSATelement(sat, bottomRight.y, bottomRight.x);
//
//       val += a + d - b - c;
//     }
//   } // end loop over disk sections
//
//   return val;
// }
//
// template <typename Tdata, typename Tidx, typename Tsection = int16_t,
//           typename Tscale = double>
// __global__ void convolve_via_SAT_and_rowSums_naive_kernel(
//     const sats::DiskRowSAT<Tsection, Tscale> disk,
//     const containers::Image<Tdata, Tidx> orig,
//     const containers::Image<Tdata, Tidx> rowSums,
//     const containers::Image<Tdata, Tidx> sat,
//     containers::Image<Tdata, Tidx> out) {
//
//   for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < (int)orig.height;
//        y += blockDim.y * gridDim.y) {
//     for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < (int)orig.width;
//          x += blockDim.x * gridDim.x) {
//
//       Tscale val =
//           sumOverDisk_SAT_and_rowSums_threadwork<Tdata, Tidx, Tsection,
//           Tscale>(
//               disk, orig, rowSums, sat, x, y);
//
//       // Value is ready here, write it back
//       out.at(y, x) = static_cast<Tdata>(val * disk.scale);
//     }
//   }
// }
//
// template <typename Tdata, typename Tidx, typename Tsection = int16_t,
//           typename Tscale = double, typename Trule>
// __global__ void convolve_via_SAT_and_rowSums_dynamicDisks_kernel(
//     const sats::DiskRowSAT<Tsection, Tscale> *disks, const Trule rule,
//     const containers::Image<Tdata, Tidx> orig,
//     const containers::Image<Tdata, Tidx> rowSums,
//     const containers::Image<Tdata, Tidx> sat,
//     containers::Image<Tdata, Tidx> out) {
//
//   for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < orig.height;
//        y += blockDim.y * gridDim.y) {
//     for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < orig.width;
//          x += blockDim.x * gridDim.x) {
//
//       // Determine the disk required
//       // Example here is to use radius-based linear steps starting from the
//       // centre to determine the disk index. Any other rule should work
//       unsigned int diskIndex = rule.getDiskIndex(y, x);
//       const sats::DiskRowSAT<Tsection, Tscale> &disk = disks[diskIndex];
//
//       // Every thread (potentially) uses its own disk
//       Tscale val =
//           sumOverDisk_SAT_and_rowSums_threadwork<Tdata, Tidx, Tsection,
//           Tscale>(
//               disk, orig, rowSums, sat, x, y);
//
//       // if (threadIdx.x == 0 && threadIdx.y == 0) {
//       //   printf(" Disk index for (row %d, col %d) is %d -> %d sections\n",
//       y,
//       //   x,
//       //          diskIndex, disk.numSections);
//       // }
//
//       // Value is ready here, write it back
//       out.at(y, x) = static_cast<Tdata>(val * disk.scale);
//     }
//   }
// }

template <typename Tdata, typename Tsection = int16_t> class DiskConvolver {
private:
  DiskConvolver(int height, int width) : m_height(height), m_width(width) {}

public:
  void resizeSections(size_t numSections) {
    m_h_sections.resize(numSections);
    m_h_sectionTypes.resize(numSections);
    m_d_sections.resize(numSections);
    m_d_sectionTypes.resize(numSections);
  }

protected:
  int m_height;
  int m_width;

  thrust::pinned_host_vector<DiskSection<Tsection>> m_h_sections;
  thrust::pinned_host_vector<uint8_t> m_h_sectionTypes;

  thrust::device_vector<DiskSection<Tsection>> m_d_sections;
  thrust::device_vector<uint8_t> m_d_sectionTypes;
};

template <typename Tdata, typename Tsection = int16_t>
class DiskConvolver_PrefixRowsSAT : public DiskConvolver<Tdata, Tsection> {
public:
  DiskConvolver_PrefixRowsSAT(int height, int width)
      : DiskConvolver<Tdata, Tsection>(height, width),
        m_d_rowSums(width * height), m_d_transpose(width * height),
        m_d_sat(width * height), m_cubwRowScan(width * height),
        m_cubwColScanTranspose(width * height) {}

  void exec(const Tdata *d_img, const sats::DiskRowSAT<Tsection> d_disk,
            Tdata *d_out, const dim3 tpb = dim3(32, 4),
            cudaStream_t stream = 0) {
    // Run pre-process to fill row sums and SAT
    preprocess(d_img, stream);

    // Wrap device arrays in containers
    containers::Image<int, unsigned int> image(d_img, this->m_width,
                                               this->m_height);
    containers::Image<int, unsigned int> rowSums(this->m_d_rowSums.data().get(),
                                                 this->m_width, this->m_height);
    containers::Image<int, unsigned int> sat(this->m_d_sat.data().get(),
                                             this->m_width, this->m_height);
    containers::Image<int, unsigned int> out(d_out, this->m_width,
                                             this->m_height);

    // Perform convolutions via lookups
    constexpr int factor = 1;
    dim3 blks((this->width + tpb.x - 1) / tpb.x,
              (this->height + tpb.y - 1) / tpb.y / factor);
    // convolve_via_SAT_and_rowSums_naive_kernel<Tdata, unsigned int, Tsection>
    //     <<<blks, tpb, 0, stream>>>(d_disk, image, rowSums, sat, out);
  }

  const thrust::device_vector<Tdata> &d_rowSums() const { return m_d_rowSums; }
  const thrust::device_vector<Tdata> &d_transpose() const {
    return m_d_transpose;
  }
  const thrust::device_vector<Tdata> &d_sat() const { return m_d_sat; }

private:
  thrust::device_vector<Tdata> m_d_rowSums;
  thrust::device_vector<Tdata> m_d_transpose;
  thrust::device_vector<Tdata> m_d_sat;

  cubw::DeviceScan::InclusiveSumByKey<
      thrust::transform_iterator<IndexToRowFunctor,
                                 thrust::counting_iterator<int>>,
      int *, int *>
      m_cubwRowScan;

  cubw::DeviceScan::InclusiveSumByKey<
      thrust::transform_iterator<IndexToRowFunctor,
                                 thrust::counting_iterator<int>>,
      int *, int *>
      m_cubwColScanTranspose;

  void preprocess(const Tdata *d_data, cudaStream_t stream = 0) {
    // CUB related prep
    auto rowKeyIterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), IndexToRowFunctor{this->m_width});
    auto colTransposeKeyIterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), IndexToRowFunctor{this->m_height});

    // 1. Inclusive prefix sums across rows
    this->m_cubwRowScan.exec(
        rowKeyIterator, d_data, this->m_d_rowSums.data().get(),
        this->m_width * this->m_height, ::cuda::std::equal_to<>(), stream);

    // 2. Inclusive prefix sums across columns
    // 2a. Transpose row sums
    transpose<int>(this->m_d_transpose.data().get(),
                   this->m_d_rowSums.data().get(), (int)this->m_height,
                   (int)this->m_width, stream);

    // 2b. Sum across rows of transpose (column sums)
    this->m_cubwColScanTranspose.exec(
        colTransposeKeyIterator, this->m_d_transpose.data().get(),
        this->m_d_transpose.data().get(), this->m_width * this->m_height,
        ::cuda::std::equal_to<>(), stream);

    // 2c. Transpose back into SAT
    transpose<int>(this->m_d_sat.data().get(), this->m_d_transpose.data().get(),
                   (int)this->m_width, (int)this->m_height, stream);
  }
};

} // namespace sats

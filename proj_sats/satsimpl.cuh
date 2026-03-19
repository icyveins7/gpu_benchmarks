#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <pinnedalloc.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
#include "transpose.cuh"

namespace sats {

enum SectionType : uint8_t {
  LOOKUP_TYPE_PIXEL,
  LOOKUP_TYPE_ROW,
  LOOKUP_TYPE_COL,
  LOOKUP_TYPE_RECT
};

std::string sectionTypeString(uint8_t type) {
  switch (type) {
  case LOOKUP_TYPE_PIXEL:
    return "PIXEL";
  case LOOKUP_TYPE_ROW:
    return "ROW";
  case LOOKUP_TYPE_COL:
    return "COL";
  case LOOKUP_TYPE_RECT:
    return "RECT";
  default:
    throw std::runtime_error("Unknown section type");
  }
}

/**
 * @brief Describes a single section of a disk.
 * All internal values like startRow/endRow are OFFSETS i.e.
 * they should begin from (roughly) -radiusPixels to +radiusPixels.
 *
 * @example
 * For a simple radius 1 disk,
 * - O -  => start/endRow = -1/-1, start/endCol = 0/0
 * O O O  => start/endRow = 0/0, start/endCol = -1/1
 * - O -  => start/endRow = 1/1, start/endCol = 0/0
 *
 * @tparam T Internal index type
 */
template <typename T = int16_t> struct DiskSection {
  static_assert(std::is_integral_v<T>,
                "T must be an integer type for ConstantDiskSection");
  static_assert(std::is_signed_v<T>,
                "T must be signed for ConstantDiskSection");

  T startRow;
  T endRow;    // inclusive
  T colOffset; // extends from -colOffset to +colOffset, since symmetric
  // T startCol;
  // T endCol; // inclusive

  __host__ __device__ T widthPixels() const { return colOffset * 2 + 1; }
  __host__ __device__ T heightPixels() const { return endRow - startRow + 1; }

  __host__ __device__ uint8_t type() const {
    if (startRow == endRow) {
      if (colOffset == 0) {
        return SectionType::LOOKUP_TYPE_PIXEL;
      } else {
        return SectionType::LOOKUP_TYPE_ROW;
      }
    } else {
      if (colOffset == 0) {
        return SectionType::LOOKUP_TYPE_COL;
      } else {
        return SectionType::LOOKUP_TYPE_RECT;
      }
    }
  }

  __host__ __device__ bool operator==(const DiskSection &other) const {
    return startRow == other.startRow && endRow == other.endRow &&
           colOffset == other.colOffset;
  }
};

/**
 * @brief Disk that can be operated using prefix sums and SAT alone.
 * This is a non-owning container, and is what is passed into kernels.
 *
 * @detail The sections are ordered in an incrementing row-order. The only types
 * that should be present are PIXEL, ROW and RECT. Sections are only specified
 * up to the middle section (which is assumed to be included, and is assumed to
 * be non-repeated, and should be a ROW).
 *
 * @example
 * For radiusPixels of 3, there are 3 sections (numSections) specified,
 * even though there are 5 sections in total (numSectionsToIterate)
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
  const DiskSection<T>
      *sections; // there should be at most (radiusPixels + 1) sections
  const uint8_t *sectionTypes; // matched with sections
  Tscale scale = 1.0;
  int radiusPixels;
  int numSections; // this is the number of *actual* sections in storage

  __host__ __device__ DiskRowSAT() {}
  __host__ __device__ DiskRowSAT(const Tscale scale, const int radiusPixels,
                                 const int numSections,
                                 const DiskSection<T> *sections,
                                 const uint8_t *sectionTypes) {
    this->sections = sections;
    this->sectionTypes = sectionTypes;
    this->scale = scale;
    this->radiusPixels = radiusPixels;
    this->numSections = numSections;
  }

  // Convenience from host_vector
  __host__ DiskRowSAT(Tscale scale, int radiusPixels, int numSections,
                      thrust::host_vector<DiskSection<T>> &sections,
                      thrust::host_vector<uint8_t> &sectionTypes) {
    this->sections = sections.data();
    this->sectionTypes = sectionTypes.data();
    this->scale = scale;
    this->radiusPixels = radiusPixels;
    this->numSections = numSections;
  }

  // Convenience from device_vector
  __host__ DiskRowSAT(Tscale scale, int radiusPixels, int numSections,
                      thrust::device_vector<DiskSection<T>> &sections,
                      thrust::device_vector<uint8_t> &sectionTypes) {
    this->sections = sections.data().get();
    this->sectionTypes = sectionTypes.data().get();
    this->scale = scale;
    this->radiusPixels = radiusPixels;
    this->numSections = numSections;
  }

  __host__ __device__ unsigned int numActivePixels() const {
    unsigned int numActive = 0;
    for (int i = 0; i < numSectionsToIterate(); i++) {
      auto section = getSection(i);
      numActive += section.heightPixels() * section.widthPixels();
    }
    return numActive;
  }

  __host__ __device__ int lengthPixels() const { return radiusPixels * 2 + 1; }
  __host__ __device__ static int lengthPixels(int radiusPixels) {
    return radiusPixels * 2 + 1;
  }
  /**
   * @brief Returns the total number of sections to iterate over
   * (as opposed to the number that is stored, which is about half).
   */
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
   * @param i Input index, 0 to numSectionsToIterate()
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
   * Note that we return a section itself, since we may have to mutate it for
   * the mirrored side.
   *
   * @param i Input index, 0 to numSectionsToIterate()
   * @return Appropriate section
   */
  __host__ __device__ const DiskSection<T> getSection(int i) const {
    bool isMirrored = i >= numSections;
    i = accessIndex(i);
    auto section = sections[i];
    // Flip the section rows
    // e.g. row -5 to -1 should now be row 1 to 5
    if (isMirrored) {
      auto startRow = -section.endRow;
      auto endRow = -section.startRow;
      section.startRow = startRow;
      section.endRow = endRow;
      // NOTE: this will correctly *not* trigger
      // when the final section is a RECT that extends past the centre;
      // only triggers for the sections after that centre section
    }
    return section;
  }

  /**
   * @brief Retrieves section type assuming a mirror around the final specified
   * section.
   * @detail Assumes input index has been validated (see validSectionIndex()).
   *
   * @param i Input index, 0 to numSectionsToIterate()
   * @return Appropriate section type
   */
  __host__ __device__ uint8_t getSectionType(int i) const {
    return sectionTypes[accessIndex(i)];
  }
};

int getMaximumSectionsForDisk_prefixRows_SAT(const double radiusPixels) {
  // rounds towards zero for the radius i.e. if radius 1.1 then only extend 1
  // pixel. +1 to account for the centre pixel. note that our disk sections
  // are mirrored so we don't store 2*radiusPixels
  return radiusPixels + 1;
}

/**
 * @brief Constructs the disk sections. Assumes pre-allocation according to
 * getMaximumSectionsForDisk_prefixRows_SAT() for sections and sectionTypes.
 */
template <typename T>
int constructSectionsForDisk_prefixRows_SAT(const double radiusPixels,
                                            DiskSection<T> *sections,
                                            uint8_t *sectionTypes) {
  int numSections = 0;

  std::vector<T> rowOffsets(
      getMaximumSectionsForDisk_prefixRows_SAT(radiusPixels));

  // Go down the rows, including the middle row
  /*
  NOTE: assumption is that for non-integer valued radius, the centre pixel is
  still on the centre row/col index exactly. e.g. for radius 1.01,
  O X O
  X C X
  O X O
  --> row 1 is still the centre

  Hence the following cast:
  */
  int pixelCentre = (int)radiusPixels;

  for (size_t i = 0; i < rowOffsets.size(); ++i) {
    double y =
        -pixelCentre + (double)i; // include offset i.e. assume centre is 0,0
    double x = std::sqrt(radiusPixels * radiusPixels - y * y); // right root

    // printf("y = %f, x = %f\n", y, x);
    rowOffsets.at(i) = (T)x; // round towards zero
  }

  // Now combine rows that have the same length i.e. offset
  DiskSection<T> section{0, 0, rowOffsets.at(0)};
  for (size_t i = 1; i < rowOffsets.size(); ++i) {
    if (rowOffsets.at(i) == rowOffsets.at(i - 1)) {
      // Change the current section
      section.endRow = i;
    } else {
      // Amend the section row values to be actual offsets
      section.startRow -= pixelCentre;
      section.endRow -= pixelCentre;
      // Push current section
      sections[numSections] = section;
      sectionTypes[numSections] = section.type();
      numSections++;
      // Make a new section
      section = {(T)i, (T)i, rowOffsets.at(i)};
    }
  }
  // Amend the section row values to be actual offsets
  section.startRow -= pixelCentre;
  section.endRow -= pixelCentre;
  // For the last section, it may be a SectionType RECT i.e. hasn't ended yet
  // Hence we must extend the rectangle to cover beyond the centre row
  section.endRow = -section.startRow; // no diff it's a SectionType ROW
  // Push last section
  sections[numSections] = section;
  sectionTypes[numSections] = section.type();
  numSections++;

  return numSections;
}

/**
 * @brief Retrieve the disk section for a given row.
 */
template <typename T>
DiskSection<T> getDiskSectionForRow(const DiskSection<T> *sections,
                                    const int numActualSections, const int row,
                                    int *idx = nullptr) {
  int targetRow = row;
  int finalRow = -sections[0].startRow;
  // int totalRows = finalRow * 2 + 1;
  if (targetRow > finalRow || targetRow < sections[0].startRow)
    throw std::runtime_error("Row out of bounds");

  // Mirrored side
  if (targetRow > 0)
    targetRow = -targetRow;

  // printf("Target row: %d from row %d\n", targetRow, row);

  for (int i = 0; i < numActualSections; ++i) {
    if (targetRow >= sections[i].startRow && targetRow <= sections[i].endRow) {
      if (idx != nullptr)
        *idx = i;
      return sections[i];
    }
  }
  throw std::runtime_error("Unable to find row?");
}

// This is passed to the kernel.
template <typename Tsection, typename Tscale = double>
struct FilterOfDisksRowSAT {
  DiskRowSAT<Tsection, Tscale> *d_disks;
  int numDisks;
};

// This is used to construct and hold the underlying RAII containers
// for all the disks and associated sections.
template <typename Tsection, typename Tscale = double>
struct FilterOfDisksRowSATCreator {
  thrust::pinned_host_vector<DiskRowSAT<Tsection, Tscale>> h_disks;
  thrust::device_vector<DiskRowSAT<Tsection, Tscale>> d_disks;
  thrust::device_vector<DiskSection<Tsection>> d_sections;
  thrust::device_vector<uint8_t> d_sectionTypes;
  thrust::pinned_host_vector<DiskSection<Tsection>> h_sections;
  thrust::pinned_host_vector<uint8_t> h_sectionTypes;

  int numDisks() const { return h_disks.size(); }

  FilterOfDisksRowSAT<Tsection, Tscale> toDevice() {
    FilterOfDisksRowSAT<Tsection, Tscale> container{d_disks.data().get(),
                                                    d_disks.size()};
    return container;
  }

  FilterOfDisksRowSATCreator(const Tscale *scale, const double *radiusPixels,
                             const int NumDisks) {

    // Count how much we would need
    int maxSections = 0;
    for (int i = 0; i < NumDisks; ++i) {
      int numSections =
          getMaximumSectionsForDisk_prefixRows_SAT(radiusPixels[i]);
      printf("Max sections for disk %d: %d\n", i, numSections);
      maxSections += numSections;
    }
    printf("Total max sections: %d\n", maxSections);
    h_sections.resize(maxSections);
    h_sectionTypes.resize(maxSections);
    // Make the disks
    std::vector<int> numActualSectionsPerDisk(NumDisks);
    int totalActualSections = 0;
    DiskSection<Tsection> *h_section_ptr = h_sections.data().get();
    uint8_t *h_sectionType_ptr = h_sectionTypes.data().get();
    for (int i = 0; i < NumDisks; ++i) {
      int numActualSections = constructSectionsForDisk_prefixRows_SAT(
          radiusPixels[i], h_section_ptr, h_sectionType_ptr);
      h_section_ptr += numActualSections;
      h_sectionType_ptr += numActualSections;
      printf("Num actual sections for disk %d: %d\n", i, numActualSections);
      totalActualSections += numActualSections;
      numActualSectionsPerDisk[i] = numActualSections; // cache this for later
    }
    // Copy to device
    d_sections.resize(totalActualSections);
    d_sectionTypes.resize(totalActualSections);
    printf("d_sections size: %d\n", d_sections.size());
    printf("d_sectionTypes size: %d\n", d_sectionTypes.size());
    // Don't really need streams here, should be a one time thing during prep
    thrust::copy(h_sections.begin(), h_sections.begin() + totalActualSections,
                 d_sections.begin());
    thrust::copy(h_sectionTypes.begin(),
                 h_sectionTypes.begin() + totalActualSections,
                 d_sectionTypes.begin());

    // Construct the wrapper containers
    int offsetToDiskSections = 0;
    h_disks.resize(NumDisks);
    d_disks.resize(NumDisks);
    for (int i = 0; i < NumDisks; ++i) {
      printf("Creating DiskRowSAT %d with sections offset %d, scale %f\n", i,
             offsetToDiskSections, scale[i]);

      // Remember, these are intended to be containers for the kernels,
      // so they house the device pointers
      h_disks[i] = DiskRowSAT<Tsection, Tscale>(
          scale[i], radiusPixels[i], numActualSectionsPerDisk[i],
          d_sections.data().get() + offsetToDiskSections,
          d_sectionTypes.data().get() + offsetToDiskSections);

      if (i < NumDisks - 1)
        offsetToDiskSections += numActualSectionsPerDisk[i];
    }
    // Copy disk containers to device vector
    thrust::copy(h_disks.begin(), h_disks.end(), d_disks.begin());
  }

  // For debugging, very helpful
  __host__ void print() {
    // NOTE: we need to create iterators here to read since
    // the disks actually contain device pointers, so we can't just access
    // them from the pinned_host_vector
    auto sectionTypePtr = h_sectionTypes.data().get();
    auto sectionPtr = h_sections.data().get();
    for (int d = 0; d < numDisks(); ++d) {
      // we read from the pinned host vector
      auto disk = h_disks[d];

      printf("Disk length %d, with %d sections, scale %f\n",
             disk.lengthPixels(), disk.numSections, disk.scale);
      for (int i = 0; i < disk.numSections; ++i) {
        printf("Section %d: type %s row %d:%d col %d:%d\n", i,
               sats::sectionTypeString(sectionTypePtr[i]).c_str(),
               sectionPtr[i].startRow, sectionPtr[i].endRow,
               -sectionPtr[i].colOffset, sectionPtr[i].colOffset);
      }

      printf("-----\n");
      for (int i = -disk.radiusPixels; i < disk.radiusPixels + 1; ++i) {
        auto section =
            sats::getDiskSectionForRow(sectionPtr, disk.numSections, i);
        printf("Row %d -> col %d : %d\n", i, -section.colOffset,
               section.colOffset);
      }
      printf("-----\n");

      // Move to offset to next disk
      sectionTypePtr += disk.numSections;
      sectionPtr += disk.numSections;
    }
  }

  int maxDiskLength() const {
    int maxLength = 0;
    for (int i = 0; i < numDisks(); ++i) {
      maxLength = std::max(maxLength, h_disks[i].lengthPixels());
    }
    return maxLength;
  }

  template <typename Tmat> __host__ std::vector<Tmat> makeMat() const {
    // check max matrix required size
    std::vector<Tmat> mat;
    int maxLength = maxDiskLength();
    mat.resize(maxLength * maxLength);
    std::memset(mat.data(), 0, maxLength * maxLength * sizeof(Tmat));
    // NOTE: in order to get the convenience of the DiskRowSAT struct
    // methods, we need to construct new ones here, since the
    // pinned_host_vector holds device pointers internally
    int offsetToDiskSections = 0;
    for (int i = 0; i < h_disks.size(); ++i) {
      auto disk = DiskRowSAT<Tsection, Tscale>(
          h_disks[i].scale, h_disks[i].radiusPixels, h_disks[i].numSections,
          h_sections.data().get() + offsetToDiskSections,
          h_sectionTypes.data().get() + offsetToDiskSections);
      offsetToDiskSections += disk.numSections;

      // use the newly created disk which points to host memory
      // to set matrix values
      for (int s = 0; s < disk.numSectionsToIterate(); ++s) {
        auto section = disk.getSection(s);
        for (int r = section.startRow; r <= section.endRow; ++r) {
          for (int c = -section.colOffset; c <= section.colOffset; ++c) {
            int y = r + maxLength / 2;
            int x = c + maxLength / 2;
            mat[y * maxLength + x] += disk.scale;
          }
        }
      }
    }
    return mat;
  }
};

// ==================================================================
// ==================================================================
// ==================================================================
// ==================================================================
// ==================================================================

// NOTE: this is just an example of a Trule to be used in dynamic filter kernel
template <typename Tidx> struct FilterSelectionRule {
  float stepSize;
  Tidx centreX;
  Tidx centreY;
  unsigned int maxDisks;

  FilterSelectionRule(float _stepSize, Tidx _centreX, Tidx _centreY,
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
struct RadialThresholdLinearGradientRule : public FilterSelectionRule<Tidx> {
  float minRadius; // disk index is 0 until this radius
  float maxRadius; // last disk index is hit at this radius

  RadialThresholdLinearGradientRule(float _stepSize, Tidx _centreX,
                                    Tidx _centreY, float _minRadius,
                                    float _maxRadius)
      : FilterSelectionRule<Tidx>(
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
    thrust::host_vector<int> &h_diskRadii, const FilterSelectionRule<T> rule) {

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

template <typename Tin, typename Trowsum, typename Tsat, typename Tidx,
          typename Tsection = int16_t, typename Tscale = double>
__device__ void sumOverDisk_SAT_and_rowSums_threadwork(
    const sats::DiskRowSAT<Tsection, Tscale> disk,
    const containers::Image<const Tin, Tidx> orig,
    const containers::Image<const Trowsum, Tidx> rowSums,
    const containers::Image<const Tsat, Tidx> sat, const int x, const int y,
    Tsat &val) {

  // NOTE: internally we use Tsat to accumulate the value first.
  // this is because Tscale is floating point (and potentially a double),
  // which is expensive to keep accumulating into.
  // It is expected that Tin, Trowsum and Tsat should be either equal
  // word-lengths or increasing word-lengths, but *they may be integers*. This
  // would entail no loss in precision when using Tsat to accumulate, but would
  // reduce the potential double-precision arithmetic at every step, which is
  // very costly. Early profiling shows that it be around 20% slower when using
  // Tscale = double to accumulate directly.
  // Example is Tin = int32, Trowsum and Tsat = int64 (to avoid overflow).
  // Then it is safe to use Tsat = int64 to accumulate the value.

  // Each thread's output value, and then loop over the sections to
  // accumulate
  for (int i = 0; i < disk.numSectionsToIterate(); ++i) {
    uint8_t sectionType = disk.getSectionType(i);
    sats::DiskSection<Tsection> section = disk.getSection(i);
    // if (x == 1 && y == 0) {
    //   printf("On section %d, type %u, val is %d\n", i, sectionType, val);
    // }

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

      Tsat satVal = a + d - b - c;
      // // DEBUG
      // if (x == 1 && y == 0) {
      //   printf("row %d, col %d, sectRow %d:%d, sectCol %d\n", y, x,
      //          section.startRow, section.endRow, section.colOffset);
      //   printf("x + colOffset = %d\n", x + section.colOffset);
      //   printf("topRight %d, %d\n", topRight.x, topRight.y);
      //   printf("%d,%d[%ld] / %d,%d[%ld] / %d,%d[%ld] / %d,%d[%ld]\n", //
      //          topLeft.y, topLeft.x, a,                               //
      //          topRight.y, topRight.x, b,                             //
      //          bottomLeft.y, bottomLeft.x, c,                         //
      //          bottomRight.y, bottomRight.x, d);                      //
      //   printf("satVal = %ld \n", satVal);
      // }
      val += satVal;
      // if (x == 1 && y == 0) {
      //   printf("val = %ld\n", val);
      // }
    }
  } // end loop over disk sections
}

/**
 * @brief Convolves entire input with a single filter (of possibly multiple
 * disks).
 *
 * @tparam Tin Input type
 * @tparam Trowsum Row sum type
 * @tparam Tsat SAT type
 * @tparam Tout Output type
 * @tparam Tidx Index type
 * @param filter Filter pointing to multiple disks, see
 * FilterOfDisksRowSATCreator
 * @param orig Input
 * @param rowSums Row sums, computed via DiskConvolver_PrefixRowsSAT
 * @param sat SAT, computed via DiskConvolver_PrefixRowsSAT
 * @param out Output
 */
template <typename Tin, typename Trowsum, typename Tsat, typename Tout,
          typename Tidx, typename Tsection = int16_t, typename Tscale = double,
          bool incrementInsteadOfSet = false>
__global__ void convolve_via_SAT_and_rowSums_naive_kernel(
    const FilterOfDisksRowSAT<Tsection, Tscale> filter,
    const containers::Image<const Tin, Tidx> orig,
    const containers::Image<const Trowsum, Tidx> rowSums,
    const containers::Image<const Tsat, Tidx> sat,
    containers::Image<Tout, Tidx> out) {

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < (int)orig.height;
       y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < (int)orig.width;
         x += blockDim.x * gridDim.x) {

      Tscale totalVal = 0;

      // Iterate over the disks in the filter
      for (int d = 0; d < filter.numDisks; ++d) {
        sats::DiskRowSAT<Tsection, Tscale> disk = filter.d_disks[d];
        // if (x == 0 && y == 0) {
        //   printf("On disk %d, has numSections %d\n", d, disk.numSections);
        // }
        Tsat val = 0;
        sumOverDisk_SAT_and_rowSums_threadwork<Tin, Trowsum, Tsat, Tidx,
                                               Tsection, Tscale>(
            disk, orig, rowSums, sat, x, y, val);

        totalVal += val * disk.scale;
      }

      // Value is ready here, write it back
      if constexpr (incrementInsteadOfSet)
        out.at(y, x) += static_cast<Tout>(totalVal);
      else
        out.at(y, x) = static_cast<Tout>(totalVal);
    }
  }
}

template <typename Tin, typename Trowsum, typename Tsat, typename Tout,
          typename Tidx, typename Trule, typename Tsection = int16_t,
          typename Tscale = double>
__global__ void convolve_via_SAT_and_rowSums_dynamicFilters_kernel(
    const sats::FilterOfDisksRowSAT<Tsection, Tscale> *filters,
    const Trule rule, const containers::Image<Tin, Tidx> orig,
    const containers::Image<Trowsum, Tidx> rowSums,
    const containers::Image<Tsat, Tidx> sat,
    containers::Image<Tout, Tidx> out) {

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < orig.height;
       y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < orig.width;
         x += blockDim.x * gridDim.x) {

      // Determine the filter required
      // This is a templated functor effectively
      unsigned int filterIndex = rule.getFilterIndex(y, x);
      const sats::FilterOfDisksRowSAT<Tsection, Tscale> filter =
          filters[filterIndex];

      // Once the filter is retrieved, everything is identical to the single
      // filter version
      Tscale totalVal = 0;

      // Iterate over the disks in the filter
      for (int d = 0; d < filter.numDisks; ++d) {
        sats::DiskRowSAT<Tsection, Tscale> disk = filter.d_disks[d];
        // if (x == 0 && y == 0) {
        //   printf("On disk %d, has numSections %d\n", d, disk.numSections);
        // }
        Tsat val = 0;
        sumOverDisk_SAT_and_rowSums_threadwork<Tin, Trowsum, Tsat, Tidx,
                                               Tsection, Tscale>(
            disk, orig, rowSums, sat, x, y, val);

        totalVal += val * disk.scale;
      }

      // Value is ready here, write it back
      out.at(y, x) = static_cast<Tout>(totalVal);
    }
  }
}

// TODO: finish this?
template <typename Tdata, typename Tsection = int16_t>
class DiskSectionContainer {
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

  DiskSectionContainer(int height, int width)
      : m_height(height), m_width(width) {}

  thrust::pinned_host_vector<DiskSection<Tsection>> m_h_sections;
  thrust::pinned_host_vector<uint8_t> m_h_sectionTypes;

  thrust::device_vector<DiskSection<Tsection>> m_d_sections;
  thrust::device_vector<uint8_t> m_d_sectionTypes;
};

template <typename Tin, typename Trowsum, typename Tsat,
          typename Tsection = int16_t>
class PrefixRowsSAT : public DiskSectionContainer<Tin, Tsection> {
public:
  PrefixRowsSAT(int height, int width)
      : DiskSectionContainer<Tin, Tsection>(height, width),
        m_d_rowSums(width, height), m_d_transpose(width, height),
        m_d_sat(width, height), m_cubwRowScan(width * height),
        m_cubwColScanTranspose(width * height) {}

  void preprocess(const Tin *d_data, cudaStream_t stream = 0) {
    // CUB related prep
    auto rowKeyIterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), IndexToRowFunctor{this->m_width});
    auto colTransposeKeyIterator = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), IndexToRowFunctor{this->m_height});

    // 1. Inclusive prefix sums across rows
    this->m_cubwRowScan.exec(
        rowKeyIterator, d_data, this->m_d_rowSums.vec.data().get(),
        this->m_width * this->m_height, ::cuda::std::equal_to<>(), stream);

    // 2. Inclusive prefix sums across columns
    // 2a. Transpose row sums
    // Since we are using in-place for the column sum, we must transpose
    // into a type Tsat array
    transpose<Trowsum, Tsat>(this->m_d_transpose.vec.data().get(),
                             this->m_d_rowSums.vec.data().get(),
                             (int)this->m_height, (int)this->m_width, stream);

    // 2b. Sum across rows of transpose (column sums)
    this->m_cubwColScanTranspose.exec(
        colTransposeKeyIterator, this->m_d_transpose.vec.data().get(),
        this->m_d_transpose.vec.data().get(), this->m_width * this->m_height,
        ::cuda::std::equal_to<>(), stream);

    // 2c. Transpose back into SAT
    transpose<Tsat, Tsat>(this->m_d_sat.vec.data().get(),
                          this->m_d_transpose.vec.data().get(),
                          (int)this->m_width, (int)this->m_height, stream);
  }

  const containers::DeviceImageStorage<Trowsum> &d_rowSums() const {
    return m_d_rowSums;
  }
  const containers::DeviceImageStorage<Tsat> &d_transpose() const {
    return m_d_transpose;
  }
  const containers::DeviceImageStorage<Tsat> &d_sat() const { return m_d_sat; }

private:
  containers::DeviceImageStorage<Trowsum> m_d_rowSums;
  containers::DeviceImageStorage<Tsat> m_d_transpose;
  containers::DeviceImageStorage<Tsat> m_d_sat;

  // input->rowsums, out-of-place prefix sum
  cubw::DeviceScan::InclusiveSumByKey<
      thrust::transform_iterator<IndexToRowFunctor,
                                 thrust::counting_iterator<int>>,
      const Tin *, Trowsum *>
      m_cubwRowScan;

  // transpose->transpose, in-place prefix sum
  cubw::DeviceScan::InclusiveSumByKey<
      thrust::transform_iterator<IndexToRowFunctor,
                                 thrust::counting_iterator<int>>,
      const Tsat *, Tsat *>
      m_cubwColScanTranspose;
};

} // namespace sats

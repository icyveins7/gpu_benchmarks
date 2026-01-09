#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

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
  DiskSection<T>
      *sections;         // there should be at most (radiusPixels + 1) sections
  uint8_t *sectionTypes; // matched with sections

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
  __host__ __device__ DiskSection<T> &getSection(int i) const {
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

} // namespace sats

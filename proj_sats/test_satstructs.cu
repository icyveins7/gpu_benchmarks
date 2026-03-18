#include "satsimpl.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

template <typename T>
int test_sectionConstruction_prefixRowsWithSAT(
    double radiusPixels, thrust::host_vector<sats::DiskSection<T>> &h_sections,
    thrust::host_vector<uint8_t> &h_sectionTypes) {
  int maxSections =
      sats::getMaximumSectionsForDisk_prefixRows_SAT(radiusPixels);
  h_sections.resize(maxSections);
  h_sectionTypes.resize(maxSections);

  int numSections = sats::constructSectionsForDisk_prefixRows_SAT<T>(
      radiusPixels, h_sections.data(), h_sectionTypes.data());

  return numSections;
}

TEST(SectionConstruction_PrefixRowsWithSAT, 3) {
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections;
  thrust::host_vector<uint8_t> h_sectionTypes;
  int numSections = test_sectionConstruction_prefixRowsWithSAT<int16_t>(
      3, h_sections, h_sectionTypes);

  ASSERT_TRUE((size_t)numSections <= h_sections.size());

  /*
  X X X 1 X X X
  X 2 2 2 2 2 X
  X 2 2 2 2 2 X
  3 3 3 3 3 3 3
  X 4 4 4 4 4 X
  X 4 4 4 4 4 X
  X X X 5 X X X
  */

  EXPECT_EQ(h_sections[0].startRow, -3);
  EXPECT_EQ(h_sections[0].endRow, -3);
  EXPECT_EQ(h_sections[0].colOffset, 0);

  EXPECT_EQ(h_sections[1].startRow, -2);
  EXPECT_EQ(h_sections[1].endRow, -1);
  EXPECT_EQ(h_sections[1].colOffset, 2);

  EXPECT_EQ(h_sections[2].startRow, 0);
  EXPECT_EQ(h_sections[2].endRow, 0);
  EXPECT_EQ(h_sections[2].colOffset, 3);

  EXPECT_EQ(numSections, 3);
}

TEST(SectionConstruction_PrefixRowsWithSAT, 4) {
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections;
  thrust::host_vector<uint8_t> h_sectionTypes;
  int numSections = test_sectionConstruction_prefixRowsWithSAT<int16_t>(
      4, h_sections, h_sectionTypes);

  ASSERT_TRUE((size_t)numSections <= h_sections.size());

  /*
  X X X X 1 X X X X
  X X 2 2 2 2 2 X X
  X 3 3 3 3 3 3 3 X
  X 3 3 3 3 3 3 3 X
  4 4 4 4 4 4 4 4 4
  X 5 5 5 5 5 5 5 X
  X 5 5 5 5 5 5 5 X
  X X 6 6 6 6 6 X X
  X X X X 7 X X X X
  */

  EXPECT_EQ(h_sections[0].startRow, -4);
  EXPECT_EQ(h_sections[0].endRow, -4);
  EXPECT_EQ(h_sections[0].colOffset, 0);

  EXPECT_EQ(h_sections[1].startRow, -3);
  EXPECT_EQ(h_sections[1].endRow, -3);
  EXPECT_EQ(h_sections[1].colOffset, 2);

  EXPECT_EQ(h_sections[2].startRow, -2);
  EXPECT_EQ(h_sections[2].endRow, -1);
  EXPECT_EQ(h_sections[2].colOffset, 3);

  EXPECT_EQ(h_sections[3].startRow, 0);
  EXPECT_EQ(h_sections[3].endRow, 0);
  EXPECT_EQ(h_sections[3].colOffset, 4);

  EXPECT_EQ(numSections, 4);
}

TEST(SectionConstruction_PrefixRowsWithSAT, 6point9) {
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections;
  thrust::host_vector<uint8_t> h_sectionTypes;
  int numSections = test_sectionConstruction_prefixRowsWithSAT<int16_t>(
      6.9, h_sections, h_sectionTypes);

  /*
  - - - 1 1 1 1 1 1 1 - - -
  - - 2 2 2 2 2 2 2 2 2 - -
  - 3 3 3 3 3 3 3 3 3 3 3 -
  4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4
  4 4 4 4 4 4 4 4 4 4 4 4 4
  - 5 5 5 5 5 5 5 5 5 5 5 -
  - - 6 6 6 6 6 6 6 6 6 - -
  - - - 7 7 7 7 7 7 7 - - -
  */

  EXPECT_EQ(h_sections[0].startRow, -6);
  EXPECT_EQ(h_sections[0].endRow, -6);
  EXPECT_EQ(h_sections[0].colOffset, 3);

  EXPECT_EQ(h_sections[1].startRow, -5);
  EXPECT_EQ(h_sections[1].endRow, -5);
  EXPECT_EQ(h_sections[1].colOffset, 4);

  EXPECT_EQ(h_sections[2].startRow, -4);
  EXPECT_EQ(h_sections[2].endRow, -4);
  EXPECT_EQ(h_sections[2].colOffset, 5);

  EXPECT_EQ(h_sections[3].startRow, -3);
  EXPECT_EQ(h_sections[3].endRow, 3);
  EXPECT_EQ(h_sections[3].colOffset, 6);

  EXPECT_EQ(numSections, 4);
}

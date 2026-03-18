#include "satsimpl.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

template <typename T>
sats::DiskRowSAT<T> test_sectionConstruction_prefixRowsWithSAT(
    double radiusPixels, thrust::host_vector<sats::DiskSection<T>> &h_sections,
    thrust::host_vector<uint8_t> &h_sectionTypes, int &numSections) {
  int maxSections =
      sats::getMaximumSectionsForDisk_prefixRows_SAT(radiusPixels);
  h_sections.resize(maxSections);
  h_sectionTypes.resize(maxSections);

  numSections = sats::constructSectionsForDisk_prefixRows_SAT<T>(
      radiusPixels, h_sections.data(), h_sectionTypes.data());

  double scale = 1.0;
  sats::DiskRowSAT<T> diskRowSAT(scale, (int)radiusPixels, numSections,
                                 h_sections, h_sectionTypes);

  return diskRowSAT;
}

TEST(SectionConstruction_PrefixRowsWithSAT, 3) {
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections;
  thrust::host_vector<uint8_t> h_sectionTypes;
  int numSections = 0;
  sats::DiskRowSAT<int16_t> diskRowSAT =
      test_sectionConstruction_prefixRowsWithSAT<int16_t>(
          3, h_sections, h_sectionTypes, numSections);

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
  EXPECT_EQ(h_sections[0].heightPixels(), 1);
  EXPECT_EQ(h_sections[0].widthPixels(), 1);

  EXPECT_EQ(h_sections[1].startRow, -2);
  EXPECT_EQ(h_sections[1].endRow, -1);
  EXPECT_EQ(h_sections[1].colOffset, 2);
  EXPECT_EQ(h_sections[1].heightPixels(), 2);
  EXPECT_EQ(h_sections[1].widthPixels(), 5);

  EXPECT_EQ(h_sections[2].startRow, 0);
  EXPECT_EQ(h_sections[2].endRow, 0);
  EXPECT_EQ(h_sections[2].colOffset, 3);
  EXPECT_EQ(h_sections[2].heightPixels(), 1);
  EXPECT_EQ(h_sections[2].widthPixels(), 7);

  EXPECT_EQ(numSections, 3);

  EXPECT_EQ(diskRowSAT.numActivePixels(), 29);
  EXPECT_EQ(diskRowSAT.numSectionsToIterate(), 5);
  // Check that the sections returned are equal up to the numSections
  for (int i = 0; i < numSections; ++i) {
    auto origSection = h_sections[i];
    auto objSection = diskRowSAT.getSection(i);

    EXPECT_EQ(origSection, objSection);
  }
  // Check that the sections returned after the middle are the flipped sections
  {
    auto section = h_sections[1];
    section.startRow = 1;
    section.endRow = 2;
    EXPECT_EQ(section, diskRowSAT.getSection(3));
  }
  {
    auto section = h_sections[0];
    section.startRow = 3;
    section.endRow = 3;
    EXPECT_EQ(section, diskRowSAT.getSection(4));
  }
}

TEST(SectionConstruction_PrefixRowsWithSAT, 4) {
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections;
  thrust::host_vector<uint8_t> h_sectionTypes;
  int numSections = 0;
  sats::DiskRowSAT<int16_t> diskRowSAT =
      test_sectionConstruction_prefixRowsWithSAT<int16_t>(
          4, h_sections, h_sectionTypes, numSections);

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
  EXPECT_EQ(h_sections[0].heightPixels(), 1);
  EXPECT_EQ(h_sections[0].widthPixels(), 1);

  EXPECT_EQ(h_sections[1].startRow, -3);
  EXPECT_EQ(h_sections[1].endRow, -3);
  EXPECT_EQ(h_sections[1].colOffset, 2);
  EXPECT_EQ(h_sections[1].heightPixels(), 1);
  EXPECT_EQ(h_sections[1].widthPixels(), 5);

  EXPECT_EQ(h_sections[2].startRow, -2);
  EXPECT_EQ(h_sections[2].endRow, -1);
  EXPECT_EQ(h_sections[2].colOffset, 3);
  EXPECT_EQ(h_sections[2].heightPixels(), 2);
  EXPECT_EQ(h_sections[2].widthPixels(), 7);

  EXPECT_EQ(h_sections[3].startRow, 0);
  EXPECT_EQ(h_sections[3].endRow, 0);
  EXPECT_EQ(h_sections[3].colOffset, 4);
  EXPECT_EQ(h_sections[3].heightPixels(), 1);
  EXPECT_EQ(h_sections[3].widthPixels(), 9);

  EXPECT_EQ(numSections, 4);

  EXPECT_EQ(diskRowSAT.numActivePixels(), 49);
  EXPECT_EQ(diskRowSAT.numSectionsToIterate(), 7);
  // Check that the sections returned are equal up to the numSections
  for (int i = 0; i < numSections; ++i) {
    auto origSection = h_sections[i];
    auto objSection = diskRowSAT.getSection(i);

    EXPECT_EQ(origSection, objSection);
  }
  // Check that the sections returned after the middle are the flipped sections
  {
    auto section = h_sections[2];
    section.startRow = 1;
    section.endRow = 2;
    EXPECT_EQ(section, diskRowSAT.getSection(4));
  }
  {
    auto section = h_sections[1];
    section.startRow = 3;
    section.endRow = 3;
    EXPECT_EQ(section, diskRowSAT.getSection(5));
  }
  {
    auto section = h_sections[0];
    section.startRow = 4;
    section.endRow = 4;
    EXPECT_EQ(section, diskRowSAT.getSection(6));
  }
}

TEST(SectionConstruction_PrefixRowsWithSAT, 6point9) {
  thrust::host_vector<sats::DiskSection<int16_t>> h_sections;
  thrust::host_vector<uint8_t> h_sectionTypes;
  int numSections = 0;
  sats::DiskRowSAT<int16_t> diskRowSAT =
      test_sectionConstruction_prefixRowsWithSAT<int16_t>(
          6.9, h_sections, h_sectionTypes, numSections);

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
  EXPECT_EQ(h_sections[0].heightPixels(), 1);
  EXPECT_EQ(h_sections[0].widthPixels(), 7);

  EXPECT_EQ(h_sections[1].startRow, -5);
  EXPECT_EQ(h_sections[1].endRow, -5);
  EXPECT_EQ(h_sections[1].colOffset, 4);
  EXPECT_EQ(h_sections[1].heightPixels(), 1);
  EXPECT_EQ(h_sections[1].widthPixels(), 9);

  EXPECT_EQ(h_sections[2].startRow, -4);
  EXPECT_EQ(h_sections[2].endRow, -4);
  EXPECT_EQ(h_sections[2].colOffset, 5);
  EXPECT_EQ(h_sections[2].heightPixels(), 1);
  EXPECT_EQ(h_sections[2].widthPixels(), 11);

  EXPECT_EQ(h_sections[3].startRow, -3);
  EXPECT_EQ(h_sections[3].endRow, 3);
  EXPECT_EQ(h_sections[3].colOffset, 6);
  EXPECT_EQ(h_sections[3].heightPixels(), 7);
  EXPECT_EQ(h_sections[3].widthPixels(), 13);

  EXPECT_EQ(numSections, 4);

  EXPECT_EQ(diskRowSAT.numActivePixels(), 145);
  EXPECT_EQ(diskRowSAT.numSectionsToIterate(), 7);
  // Check that the sections returned are equal up to the numSections
  for (int i = 0; i < numSections; ++i) {
    auto origSection = h_sections[i];
    auto objSection = diskRowSAT.getSection(i);

    EXPECT_EQ(origSection, objSection);
  }
  // Check that the sections returned after the middle are the flipped sections
  {
    auto section = h_sections[2];
    section.startRow = 4;
    section.endRow = 4;
    EXPECT_EQ(section, diskRowSAT.getSection(4));
  }
  {
    auto section = h_sections[1];
    section.startRow = 5;
    section.endRow = 5;
    EXPECT_EQ(section, diskRowSAT.getSection(5));
  }
  {
    auto section = h_sections[0];
    section.startRow = 6;
    section.endRow = 6;
    EXPECT_EQ(section, diskRowSAT.getSection(6));
  }
}

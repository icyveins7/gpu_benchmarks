#include "containers/bitset.cuh"
#include "gtest/gtest.h"
#include <stdexcept>

TEST(Bitset, HostsideConstructors) {
  int numBits = 48;
  thrust::host_vector<unsigned int> h_data(numBits / 32 +
                                           (numBits % 32 > 0 ? 1 : 0));

  containers::Bitset<unsigned int, int> bitset(h_data, numBits);
  // Basic checks after constructor
  EXPECT_EQ(numBits, bitset.numBits);
  EXPECT_EQ(h_data.size(), bitset.numDataElements);
  EXPECT_EQ(bitset.data, h_data.data());

  // Create invalid sizes
  EXPECT_THROW((containers::Bitset<unsigned int, int>(h_data, 65)),
               std::invalid_argument);

  // Create invalid size with direct pointer and then test validity
  containers::Bitset<unsigned int, int> bitsetinvalid(h_data.data(),
                                                      h_data.size(), 65);
  EXPECT_FALSE(bitsetinvalid.hasValidNumBits());

  // Test similarly for device vector
  thrust::device_vector<unsigned int> d_data(numBits / 32 +
                                             (numBits % 32 > 0 ? 1 : 0));

  containers::Bitset<unsigned int, int> bitset2(d_data, numBits);
  EXPECT_EQ(numBits, bitset2.numBits);
  EXPECT_EQ(d_data.size(), bitset2.numDataElements);
  EXPECT_EQ(bitset2.data, d_data.data().get());

  // Create invalid sizes
  EXPECT_THROW((containers::Bitset<unsigned int, int>(d_data, 65)),
               std::invalid_argument);

  // Create invalid size with direct pointer and then test validity
  containers::Bitset<unsigned int, int> d_bitsetinvalid(d_data.data().get(),
                                                        d_data.size(), 65);
  EXPECT_FALSE(bitsetinvalid.hasValidNumBits());
}

TEST(Bitset, HostsideMethods) {
  int numBits = 48;
  thrust::host_vector<unsigned int> h_data(numBits / 32 +
                                           (numBits % 32 > 0 ? 1 : 0));

  containers::Bitset<unsigned int, int> bitset(h_data, numBits);

  // Check all 0s
  for (int i = 0; i < bitset.numBits; ++i)
    EXPECT_EQ(0, bitset.getBitAt(i));

  // Set 1 bit at a time and check
  for (int i = 0; i < bitset.numBits; ++i)
    bitset.setBitAt(i, 1);
  for (int i = 0; i < bitset.numBits; ++i)
    EXPECT_EQ(1, bitset.getBitAt(i));

  // Unset and check again
  for (int i = 0; i < bitset.numBits; ++i)
    bitset.setBitAt(i, 0);
  for (int i = 0; i < bitset.numBits; ++i)
    EXPECT_EQ(0, bitset.getBitAt(i));

  // Test out of bounds checkers
  EXPECT_FALSE(bitset.isValidBitIndex(49));
  EXPECT_FALSE(bitset.isValidElementIndex(h_data.size()));
}

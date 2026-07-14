// This is used as a replacement for GTest::gtest_main when building the tests
// primarily because Windows builds fail a lot with the default
// GTest::gtest_main. I'm not going to bother figuring out why. This works for
// now.

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#include "containers/pinnedbuffer.cuh"
#include "gtest/gtest.h"

TEST(PinnedHostBuffer, DefaultConstructor) {
  containers::PinnedHostBuffer<int> buf;
  EXPECT_EQ(nullptr, buf.data());
  EXPECT_EQ(0u, buf.size());
  EXPECT_EQ(0u, buf.capacity());
}

TEST(PinnedHostBuffer, SizeConstructor) {
  containers::PinnedHostBuffer<int> buf(16);
  EXPECT_NE(nullptr, buf.data());
  EXPECT_EQ(16u, buf.size());
  EXPECT_EQ(16u, buf.capacity());
}

TEST(PinnedHostBuffer, DataReadWrite) {
  const size_t N = 8;
  containers::PinnedHostBuffer<int> buf(N);
  for (size_t i = 0; i < N; ++i)
    buf.data()[i] = static_cast<int>(i * 2);
  for (size_t i = 0; i < N; ++i)
    EXPECT_EQ(static_cast<int>(i * 2), buf.data()[i]);
}

TEST(PinnedHostBuffer, ConstData) {
  containers::PinnedHostBuffer<int> buf(4);
  buf.data()[0] = 42;
  const containers::PinnedHostBuffer<int> &cbuf = buf;
  EXPECT_EQ(42, cbuf.data()[0]);
}

TEST(PinnedHostBuffer, ResizeFromDefaultConstructed) {
  containers::PinnedHostBuffer<int> buf;
  buf.resize(8);
  EXPECT_EQ(8u, buf.size());
  EXPECT_GE(buf.capacity(), 8u);
  EXPECT_NE(nullptr, buf.data());
}

TEST(PinnedHostBuffer, ResizeWithinCapacity) {
  containers::PinnedHostBuffer<int> buf(16);
  int *orig_ptr = buf.data();
  buf.resize(8);
  EXPECT_EQ(8u, buf.size());
  EXPECT_EQ(16u, buf.capacity()); // no realloc
  EXPECT_EQ(orig_ptr, buf.data());
}

TEST(PinnedHostBuffer, At) {
  containers::PinnedHostBuffer<int> buf(4);
  for (size_t i = 0; i < 4; ++i)
    buf.at(i) = static_cast<int>(i + 10);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(static_cast<int>(i + 10), buf.at(i));
}

TEST(PinnedHostBuffer, AtConst) {
  containers::PinnedHostBuffer<int> buf(4);
  buf.at(0) = 99;
  const containers::PinnedHostBuffer<int> &cbuf = buf;
  EXPECT_EQ(99, cbuf.at(0));
}

TEST(PinnedHostBuffer, AtOutOfRange) {
  containers::PinnedHostBuffer<int> buf(4);
  EXPECT_THROW(buf.at(4), std::out_of_range);
}

TEST(PinnedHostBuffer, ResizeBeyondCapacityPreservesData) {
  const size_t initial = 4;
  containers::PinnedHostBuffer<int> buf(initial);
  for (size_t i = 0; i < initial; ++i)
    buf.data()[i] = static_cast<int>(i + 1);

  buf.resize(16);
  EXPECT_EQ(16u, buf.size());
  EXPECT_GE(buf.capacity(), 16u);

  // data up to old size should be intact after realloc
  for (size_t i = 0; i < initial; ++i)
    EXPECT_EQ(static_cast<int>(i + 1), buf.data()[i]);
}

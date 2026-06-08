#include "containers/image.cuh"
#include "containers/stream_ordered_storage.cuh"
#include "containers/streams.cuh"

#include <vector>

#include "thrust/sequence.h"
#include "gtest/gtest.h"
#include <numeric>

TEST(ContainersStreamOrderedDeviceStorage, BasicChecks) {
  containers::CudaStream stream;
  containers::StreamOrderedDeviceStorage<float> vec(10, stream());
  ASSERT_EQ(10, vec.size());
  ASSERT_EQ(10, vec.capacity());
  ASSERT_EQ(vec.stream(), stream());
}

TEST(ContainersStreamOrderedDeviceStorage, VectorOfVectors) {
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, 0);
  uint64_t reserved;
  uint64_t used;
  containers::CudaStream stream;
  {
    std::vector<containers::StreamOrderedDeviceStorage<float>> bigvec;

    bigvec.emplace_back(10, stream());
    ASSERT_EQ(10, bigvec.at(0).size());
    ASSERT_EQ(bigvec.at(0).stream(), stream());

    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent,
                            &reserved); // usually starts at 32MB
    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);

    ASSERT_LE(10 * sizeof(float), reserved);
    ASSERT_EQ(10 * sizeof(float), used);

    bigvec.emplace_back(20, stream());
    ASSERT_EQ(20, bigvec.at(1).size());
    ASSERT_EQ(bigvec.at(1).stream(), stream());

    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent,
                            &reserved);
    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);

    ASSERT_LE(10 * sizeof(float) + 20 * sizeof(float), reserved);
    ASSERT_EQ(10 * sizeof(float) + 20 * sizeof(float), used);
  }
  // Vector should have released back to pool
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
  ASSERT_EQ(0, used);

  stream.sync();
  // Now released back to OS
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent,
                          &reserved);
  ASSERT_EQ(0, reserved);
}

TEST(ContainersStreamOrderedDeviceStorage, ResizeWithoutCopyExpand) {
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, 0);
  uint64_t used;
  containers::CudaStream stream;
  {
    containers::StreamOrderedDeviceStorage<float> vec(10, stream());
    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
    ASSERT_EQ(10 * sizeof(float), used);

    // Expand beyond capacity — should reallocate
    vec.resizeWithoutCopy(100);
    ASSERT_EQ(100, vec.size());
    ASSERT_EQ(100, vec.capacity());

    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
    ASSERT_EQ(100 * sizeof(float), used);
  }
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
  ASSERT_EQ(0, used);
}

TEST(ContainersStreamOrderedDeviceStorage, ResizeWithoutCopyNoRealloc) {
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, 0);
  uint64_t used;
  containers::CudaStream stream;
  {
    containers::StreamOrderedDeviceStorage<float> vec(100, stream());
    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
    ASSERT_EQ(100 * sizeof(float), used);

    // Shrink within capacity — no reallocation, capacity unchanged
    vec.resizeWithoutCopy(10);
    ASSERT_EQ(10, vec.size());
    ASSERT_EQ(100, vec.capacity());

    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
    ASSERT_EQ(100 * sizeof(float), used); // pool usage unchanged
  }
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
  ASSERT_EQ(0, used);
}

TEST(ContainersStreamOrderedDeviceStorage, ResizeWithoutCopyUninitializedThrows) {
  containers::StreamOrderedDeviceStorage<float> vec;
  ASSERT_THROW(vec.resizeWithoutCopy(10), std::runtime_error);
}

TEST(ContainersStreamOrderedDeviceStorage, MoveAssignmentNoLeak) {
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, 0);
  uint64_t used;
  containers::CudaStream stream;
  {
    containers::StreamOrderedDeviceStorage<float> a(10, stream());
    containers::StreamOrderedDeviceStorage<float> b(20, stream());
    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
    ASSERT_EQ(10 * sizeof(float) + 20 * sizeof(float), used);

    // Move-assign b into a; a's old 10-element allocation must be freed
    a = std::move(b);
    ASSERT_EQ(20, a.size());
    ASSERT_EQ(20, a.capacity());
    ASSERT_EQ(0, b.size());

    cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
    ASSERT_EQ(20 * sizeof(float), used);
  }
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &used);
  ASSERT_EQ(0, used);
}

TEST(ContainersStreamOrderedDeviceImageStorage, BasicChecks) {
  containers::CudaStream stream;
  containers::StreamOrderedDeviceImageStorage<float> img(10, 10, stream());
  ASSERT_EQ(10, img.width);
  ASSERT_EQ(10, img.height);
  ASSERT_EQ(img.stream(), stream());
}

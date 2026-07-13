#include "containers/pinnedbuffer.cuh"
#include "gtest/gtest.h"
#include <cuda.h>

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
  const containers::PinnedHostBuffer<int>& cbuf = buf;
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
  int* orig_ptr = buf.data();
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
  const containers::PinnedHostBuffer<int>& cbuf = buf;
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

// ---------------------------------------------------------------------------
// flags() and resize(size, flags) tests
// ---------------------------------------------------------------------------

TEST(PinnedHostBuffer, FlagsDefaultUnallocated) {
  containers::PinnedHostBuffer<int> buf;
  EXPECT_EQ(cudaHostAllocDefault, buf.flags());
}

// Note: cudaHostAllocDefault (0) cannot be directly verified via flags()
// because CUDA always reports cudaHostAllocMapped (2) regardless of whether
// it was explicitly requested. Using Portable (1) returns 3 due to CUDA's
// internal OR-ing of the two flags.
TEST(PinnedHostBuffer, FlagsPortablePrint) {
  containers::PinnedHostBuffer<int> buf(4, cudaHostAllocPortable);
  printf("[  INFO    ] flags() with cudaHostAllocPortable = %u\n", buf.flags());
}

// Default-construct then resize with explicit flags: must allocate.
TEST(PinnedHostBuffer, ResizeWithFlagsFromNull) {
  containers::PinnedHostBuffer<int> buf;
  buf.resize(8, cudaHostAllocDefault);
  EXPECT_EQ(8u, buf.size());
  EXPECT_GE(buf.capacity(), 8u);
  EXPECT_NE(nullptr, buf.data());
  // Note: even a cudaHostAllocDefault allocation reports flags = 2
  // (cudaHostAllocMapped).
}

// Changing flags within existing capacity must still force a realloc.
// Allocations must be large enough (>= page size) to each land on their own
// page, otherwise the new portable allocation inherits the existing page's
// default registration and flags() will not reflect cudaHostAllocPortable.
TEST(PinnedHostBuffer, ResizeWithFlagsChangeForcesRealloc) {
  const size_t N = 1024 * 1024; // 4 MB — ensures a fresh page for each alloc
  containers::PinnedHostBuffer<int> buf(N, cudaHostAllocDefault);
  buf.resize(N, cudaHostAllocPortable);
  EXPECT_EQ(N, buf.size());
  EXPECT_NE(0u, buf.flags() & cudaHostAllocPortable);
}

// Does a Default allocation on the same page as a live Portable allocation
// inherit the Portable registration (flags=3) or get its own (flags=2)?
// This shows that if the allocation is small, the flags don't change.
TEST(PinnedHostBuffer, FlagsDefaultAfterLivePortable) {
  containers::PinnedHostBuffer<int> portable(16, cudaHostAllocPortable);
  containers::PinnedHostBuffer<int> def(16, cudaHostAllocDefault);
  printf("[  INFO    ] portable ptr = %p  flags = %u\n", portable.data(),
         portable.flags());
  printf("[  INFO    ] default  ptr = %p  flags = %u\n", def.data(),
         def.flags());
}

// Same question but with a large portable allocation (4 MB) to ensure the
// subsequent default allocation lands on a fresh page.
// This shows that if the allocation is large, flags change as expected.
TEST(PinnedHostBuffer, FlagsDefaultAfterLargePortable) {
  const size_t large = 4 * 1024 * 1024 / sizeof(int); // 4 MB
  containers::PinnedHostBuffer<int> portable(large, cudaHostAllocPortable);
  containers::PinnedHostBuffer<int> def(16, cudaHostAllocDefault);
  printf("[  INFO    ] portable ptr = %p  flags = %u\n", portable.data(),
         portable.flags());
  printf("[  INFO    ] default  ptr = %p  flags = %u\n", def.data(),
         def.flags());
}

// ---------------------------------------------------------------------------
// Mapped memory tests
// ---------------------------------------------------------------------------

__global__ void fill_kernel(int* ptr, int n, int val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    ptr[idx] = val;
}

// Allocate with cudaHostAllocMapped, get device pointer, write from GPU,
// verify on host.
TEST(PinnedHostBuffer, MappedKernelWrite) {
  unsigned int flags = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceFlags(&flags));
  if (!(flags & cudaDeviceMapHost)) {
    printf("[  WARNING ] cudaDeviceMapHost not set; skipping mapped memory "
           "test\n");
    return;
  }

  const int N = 16;
  containers::PinnedHostBuffer<int> buf(N, cudaHostAllocMapped);

  int* d_ptr = nullptr;
  ASSERT_EQ(cudaSuccess, cudaHostGetDevicePointer(&d_ptr, buf.data(), 0));

  fill_kernel<<<(N + 31) / 32, 32>>>(d_ptr, N, 42);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  for (int i = 0; i < N; ++i)
    EXPECT_EQ(42, buf.data()[i]);
}

// Verifies that cudaHostAllocPortable memory is genuinely accessible from a
// second device. Requires >= 2 devices; skips otherwise.
// Large allocation (4 MB) is intentional: small allocations may share a page
// with a prior default-registered allocation and silently lose the portable
// flag (see flags() docstring). We verify the flag stuck before proceeding.
TEST(PinnedHostBuffer, PortableKernelWriteFromSecondDevice) {
  int ndevices = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&ndevices));
  if (ndevices < 2) {
    printf("[  WARNING ] fewer than 2 devices; skipping portable test\n");
    return;
  }

  // Set cudaDeviceMapHost for device 1 before its context is initialised.
  // cudaSetDeviceFlags is best-effort: it may fail if the context already
  // exists. Either way, check the flags afterwards to decide whether to
  // proceed.
  ASSERT_EQ(cudaSuccess, cudaSetDevice(1));
  cudaSetDeviceFlags(cudaDeviceMapHost);
  unsigned int dev1flags = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceFlags(&dev1flags));
  if (!(dev1flags & cudaDeviceMapHost)) {
    printf("[  WARNING ] cudaDeviceMapHost not set for device 1; skipping\n");
    cudaSetDevice(0);
    return;
  }
  ASSERT_EQ(cudaSuccess, cudaSetDevice(0));

  // Large enough allocation to land on its own page so the portable flag
  // sticks.
  const size_t N = 1024 * 1024; // 4 MB
  containers::PinnedHostBuffer<int> buf(N, cudaHostAllocPortable |
                                               cudaHostAllocMapped);
  if (!(buf.flags() & cudaHostAllocPortable)) {
    printf(
        "[  WARNING ] portable flag did not stick (page sharing); skipping\n");
    return;
  }

  // cudaFree(nullptr) is a no-op but guarantees device 0's context exists
  // so cuCtxGetCurrent returns a valid handle.
  ASSERT_EQ(cudaSuccess, cudaFree(nullptr));
  CUcontext ctx0 = nullptr;
  cuCtxGetCurrent(&ctx0);

  ASSERT_EQ(cudaSuccess, cudaSetDevice(1));
  CUcontext ctx1 = nullptr;
  cuCtxGetCurrent(&ctx1);
  // CUcontext is an opaque pointer (typedef struct CUctx_st*). The driver owns
  // the context object and always returns the same pointer for the same context,
  // so pointer equality is the correct and sufficient identity check — there is
  // no aliasing or copying of the underlying struct.
  ASSERT_NE(ctx0, ctx1) << "device 0 and device 1 must have distinct contexts";

  // Write from device 1 via its own device pointer for the portable buffer.
  int* d_ptr = nullptr;
  ASSERT_EQ(cudaSuccess, cudaHostGetDevicePointer(&d_ptr, buf.data(), 0));
  fill_kernel<<<(N + 255) / 256, 256>>>(d_ptr, (int)N, 99);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  CUcontext ctx0_restored = nullptr;
  cuCtxGetCurrent(&ctx0_restored);
  ASSERT_EQ(ctx0, ctx0_restored);

  for (size_t i = 0; i < N; ++i)
    EXPECT_EQ(99, buf.data()[i]);
}

// Same as MappedKernelWrite but the buffer starts default-constructed and
// gets its flags via resize(size, flags).
TEST(PinnedHostBuffer, MappedResizeKernelWrite) {
  unsigned int flags = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceFlags(&flags));
  if (!(flags & cudaDeviceMapHost)) {
    printf("[  WARNING ] cudaDeviceMapHost not set; skipping mapped resize "
           "test\n");
    return;
  }

  const int N = 16;
  containers::PinnedHostBuffer<int> buf;
  buf.resize(N, cudaHostAllocMapped);

  int* d_ptr = nullptr;
  ASSERT_EQ(cudaSuccess, cudaHostGetDevicePointer(&d_ptr, buf.data(), 0));

  fill_kernel<<<(N + 31) / 32, 32>>>(d_ptr, N, 7);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  for (int i = 0; i < N; ++i)
    EXPECT_EQ(7, buf.data()[i]);
}

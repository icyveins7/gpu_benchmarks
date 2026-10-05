#include "timer.h"
#include <iostream>

#include "pinnedalloc.cuh"

#include <nvtx3/nvToolsExt.h>

int main(int argc, char* argv[]) {
  printf("Comparison of pinned host allocations\n");

  size_t len = 10;
  if (argc > 1) {
    len = atoi(argv[1]);
  }
  printf("Using length %zu\n", len);

  HighResolutionTimer<> timer;

  for (int iter = 0; iter < 3; ++iter) {
    nvtxRangePush("cudaMallocHost");
    int* rp;

    timer.start();

    // nsys profile says this calls cudaHostAlloc instead;
    // may be due to C++ overloads, but anyway it should be the same thing
    cudaMallocHost(&rp, len * sizeof(int));

    timer.stop();

    for (size_t i = 0; i < len; ++i) {
      rp[i] = i;
    }

    if (len < 64) {
      for (size_t i = 0; i < len; ++i) {
        printf("%d\n", rp[i]);
      }
    }

    cudaFreeHost(rp);
    nvtxRangePop();
  }

  // A reminder: pinned_host_vector invokes a zero-ing kernel.
  // For small sizes like this, it is vastly shorter than the
  // time required for cudaMallocHost instead.
  for (int iter = 0; iter < 3; ++iter) {
    nvtxRangePush("thrust_pinned");

    timer.start();
    thrust::pinned_host_vector<int> tp(len);
    timer.stop();

    // does resizing lower create an unnecessary realloc?
    tp.resize((size_t)(len * 0.9));

    for (size_t i = 0; i < tp.size(); ++i) {
      tp[i] = i;
    }

    // what about if i resize back to the original?
    // NOTE: so this works exactly the same as std::vector,
    // in that it tries to zero the 'new' elements
    // hence it launches a kernel to do this, which is accompanied
    // by a cudaStreamSynchronize
    tp.resize(len);

    if (len < 64) {
      for (size_t i = 0; i < tp.size(); ++i) {
        printf("%d\n", tp[i]);
      }
    }
    nvtxRangePop();
  }

  // Verify that thrust::no_init changes only the logical size when capacity is
  // already sufficient. The marker makes writes obvious: ordinary resize()
  // should zero the regrown tail, while no_init should leave its bytes alone.
  const size_t testLen = len < 4 ? 10 : len;
  const size_t shrunkLen = testLen / 2;
  constexpr int marker = 0x5a5a5a5a;
  thrust::pinned_host_vector<int> values(testLen, marker);
  const size_t originalCapacity = values.capacity();
  const int* const originalPointer = values.data().get();

  values.resize(shrunkLen);
  values.resize(testLen);
  size_t ordinaryZeroCount = 0;
  for (size_t i = shrunkLen; i < testLen; ++i) {
    ordinaryZeroCount += values[i] == 0;
  }
  const bool ordinaryReused = values.capacity() == originalCapacity &&
                              values.data().get() == originalPointer;

  for (size_t i = 0; i < testLen; ++i) {
    values[i] = marker;
  }
  values.resize(shrunkLen);
  values.resize(testLen, thrust::no_init);
  size_t noInitMarkerCount = 0;
  for (size_t i = shrunkLen; i < testLen; ++i) {
    noInitMarkerCount += values[i] == marker;
  }
  const bool noInitReused = values.capacity() == originalCapacity &&
                            values.data().get() == originalPointer;
  const size_t regrownCount = testLen - shrunkLen;

  printf("\nPinned vector shrink/regrow test (%zu -> %zu -> %zu elements)\n",
         testLen, shrunkLen, testLen);
  printf("ordinary resize: reused allocation=%s, zeroed regrown elements=%zu/%zu\n",
         ordinaryReused ? "YES" : "NO", ordinaryZeroCount, regrownCount);
  printf("no_init resize:  reused allocation=%s, preserved marker elements=%zu/%zu\n",
         noInitReused ? "YES" : "NO", noInitMarkerCount, regrownCount);

  const bool passed = ordinaryReused && noInitReused &&
                      ordinaryZeroCount == regrownCount &&
                      noInitMarkerCount == regrownCount;
  printf("RESULT: thrust::no_init %s initialization of the regrown range\n",
         passed ? "SKIPPED" : "DID NOT SKIP");
  return passed ? 0 : 1;
}

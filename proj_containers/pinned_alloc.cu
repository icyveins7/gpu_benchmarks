#include <iostream>

#include "pinnedalloc.cuh"

#include <nvtx3/nvToolsExt.h>

int main() {
  printf("Comparison of pinned host allocations\n");

  size_t len = 10;

  for (int iter = 0; iter < 3; ++iter) {
    nvtxRangePush("cudaMallocHost");
    int *rp;

    // nsys profile says this calls cudaHostAlloc instead;
    // may be due to C++ overloads, but anyway it should be the same thing
    cudaMallocHost(&rp, len * sizeof(int));

    for (size_t i = 0; i < len; ++i) {
      rp[i] = i;
    }

    for (size_t i = 0; i < len; ++i) {
      printf("%d\n", rp[i]);
    }

    cudaFreeHost(rp);
    nvtxRangePop();
  }

  // A reminder: pinned_host_vector invokes a zero-ing kernel.
  // For small sizes like this, it is vastly shorter than the
  // time required for cudaMallocHost instead.
  for (int iter = 0; iter < 3; ++iter) {
    nvtxRangePush("thrust_pinned");
    thrust::pinned_host_vector<int> tp(len);
    for (size_t i = 0; i < len; ++i) {
      tp[i] = i;
    }

    for (size_t i = 0; i < len; ++i) {
      printf("%d\n", tp[i]);
    }
    nvtxRangePop();
  }

  return 0;
}

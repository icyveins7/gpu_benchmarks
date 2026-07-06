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

  return 0;
}

/*
Timings show that sorting is generally not worth it until you approach
maybe 10% or more of the output. Before that, it costs too much to sort as
compared to just writing directly.

Using 64-bit indices as opposed to 32-bit doesn't make much difference (at
least, much smaller than 2x). Probably all timings are dominated by the writes,
as opposed to the reads, since those are the non-coalesced ones.

Note that using a struct that is not coalesced doesn't make much difference,
since the outputs are still uncoalesced. Effectively since there is so much warp
divergence anyway the reduced read speeds appear to be insignificant.
*/

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

template <typename T, typename U>
void printResult(const T *h_in, const U *h_out, size_t inLength,
                 size_t outLength) {
  // print the input first
  for (size_t i = 0; i < inLength; ++i) {
    std::cout << h_in[i] << " ";
  }
  printf("\n");

  for (size_t i = 0; i < outLength / 100 + (outLength % 100 > 0 ? 1 : 0); ++i) {
    for (int j = 0; j < 100; ++j) {
      size_t idx = i * 100 + j;
      if (idx < outLength) {
        printf("%hhu", h_out[idx]);
      }
    }
    printf("\n");
  }
  printf("-----------------\n");
}

template <typename Tin, typename Tout>
__global__ void inflate_binary_map(const Tin *d_in, const int num_in,
                                   Tout *d_out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_in;
       i += blockDim.x * gridDim.x) {
    d_out[d_in[i]] = 1;
  }
}

template <typename Tin> struct UncoalescedIndices {
  Tin idx;
  uint64_t a;
  uint64_t b;
  uint64_t c;
  uint64_t d;
};

template <typename Tin, typename Tout>
__global__ void
inflate_binary_map_from_struct(const UncoalescedIndices<Tin> *d_in,
                               const int num_in, Tout *d_out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_in;
       i += blockDim.x * gridDim.x) {
    d_out[d_in[i].idx] = 1;
  }
}

int main(int argc, char **argv) {
  printf("Testing inflation from lists.\n");
  if (argc != 3) {
    printf("Usage: inflate <outLength> <inLength>\n");
    return 1;
  }
  size_t outLength = atoi(argv[1]);
  size_t inLength = atoi(argv[2]);

  typedef int64_t Tin;
  typedef uint8_t Tout;

  thrust::host_vector<Tin> h_in(outLength);
  thrust::sequence(h_in.begin(), h_in.end());
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(h_in.begin(), h_in.end(), g);
  thrust::device_vector<Tin> d_in(inLength);
  thrust::copy(h_in.begin(), h_in.begin() + inLength, d_in.begin());

  thrust::device_vector<Tout> d_out(outLength);
  int blks = inLength / 128 + (inLength % 128 > 0 ? 1 : 0);
  blks /= 2; // follow thrust and use a half-grid
  inflate_binary_map<<<blks, 128>>>(d_in.data().get(), inLength,
                                    d_out.data().get());

  thrust::host_vector<Tout> h_out = d_out;
  if (outLength < 1000) {
    printResult(h_in.data(), h_out.data(), inLength, outLength);
  }

  // Sort first
  thrust::fill(d_out.begin(), d_out.end(), 0); // reset
  thrust::sort(d_in.begin(), d_in.end());
  h_in = d_in;

  // Do again
  inflate_binary_map<<<blks, 128>>>(d_in.data().get(), inLength,
                                    d_out.data().get());
  h_out = d_out;
  if (outLength < 1000) {
    printResult(h_in.data(), h_out.data(), inLength, outLength);
  }

  // ========== Try on uncoalesced struct
  thrust::fill(d_out.begin(), d_out.end(), 0); // reset
  printf("Size of uncoalesced struct: %lu\n", sizeof(UncoalescedIndices<Tin>));
  thrust::host_vector<UncoalescedIndices<Tin>> h_in_uncoalesced(inLength);
  for (size_t i = 0; i < inLength; ++i) {
    h_in_uncoalesced[i].idx = h_in[i];
  }
  thrust::device_vector<UncoalescedIndices<Tin>> d_in_uncoalesced(inLength);
  thrust::copy(h_in_uncoalesced.begin(), h_in_uncoalesced.end(),
               d_in_uncoalesced.begin());
  inflate_binary_map_from_struct<<<blks, 128>>>(d_in_uncoalesced.data().get(),
                                                inLength, d_out.data().get());

  h_out = d_out;
  if (outLength < 1000) {
    printResult(h_in.data(), h_out.data(), inLength, outLength);
  }

  // ======= Try using thrust scatter
  thrust::fill(d_out.begin(), d_out.end(), 0); // reset
  nvtxRangePushA("thrust_scatter");
  thrust::scatter(thrust::make_constant_iterator<Tout>(1),
                  thrust::make_constant_iterator<Tout>(1) + d_in.size(),
                  d_in.begin(), // indices map
                  d_out.begin());
  nvtxRangePop();

  h_out = d_out;
  if (outLength < 1000) {
    printResult(h_in.data(), h_out.data(), inLength, outLength);
  }

  return 0;
}

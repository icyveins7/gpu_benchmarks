#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cstdlib>
#include <cuda/std/utility>
#include <iostream>
#include <type_traits>

template <typename Tidx, typename Pair>
__device__ void gather_pair(const Tidx *d_idx, const Tidx length, Pair d_pair) {
  static_assert(std::is_pointer<typename Pair::first_type>::value,
                "dst must be a pointer");
  static_assert(std::is_pointer<typename Pair::second_type>::value,
                "src must be a pointer");
  using Tdst =
      std::remove_cv_t<std::remove_pointer_t<typename Pair::first_type>>;
  using Tsrc =
      std::remove_cv_t<std::remove_pointer_t<typename Pair::second_type>>;
  static_assert(std::is_same<Tdst, Tsrc>::value,
                "dst and src must be the same underlying type");

  for (Tidx i = blockDim.x * blockIdx.x + threadIdx.x; i < length;
       i += blockDim.x * gridDim.x) {
    Tidx srcIdx = d_idx[i];
    d_pair.first[i] = d_pair.second[srcIdx];
  }
}

template <typename Tidx, typename... Pairs>
__global__ void gather_kernel(const Tidx *d_idx, const Tidx length,
                              Pairs... d_pairs) {
  constexpr size_t num_pairs = sizeof...(d_pairs);
  static_assert(num_pairs > 0, "At least one dst/src pair must be provided");
  // separate loops for separate pairs
  ((gather_pair<Tidx>(d_idx, length, d_pairs)), ...);

  // // one loop for all pairs (read index once, but maybe cache less friendly?)
  // for (Tidx i = blockDim.x * blockIdx.x + threadIdx.x; i < length;
  //      i += blockDim.x * gridDim.x) {
  //   Tidx srcIdx = d_idx[i];
  //   ((d_pairs.first[i] = d_pairs.second[srcIdx]), ...);
  // }
}

int main(int argc, char *argv[]) {
  size_t len = 10;
  if (argc > 1) {
    len = std::atoi(argv[1]);
  }
  printf("Using length %zu\n", len);

  thrust::host_vector<int> h_in(len);
  for (size_t i = 0; i < len; ++i) {
    h_in[i] = std::rand() % 100;
  }
  if (len < 20) {
    printf("Original \n");
    for (size_t i = 0; i < len; ++i) {
      printf("%02d ", h_in[i]);
    }
    printf("\n");
  }
  thrust::device_vector<int> d_in = h_in;
#if defined(GATHER_TWO_ARRAYS)
  thrust::device_vector<int> d_in2 = d_in;
#endif

  thrust::host_vector<int> h_idx(len);
  thrust::sequence(h_idx.begin(), h_idx.end());
  thrust::sort_by_key(h_in.begin(), h_in.end(), h_idx.begin());
  if (len < 20) {
    printf("thrust sort_by_key on host\n");
    for (size_t i = 0; i < len; ++i) {
      printf("%02d ", h_in[i]);
    }
    printf("\n");
  }

  // custom kernel
  thrust::device_vector<int> d_idx = h_idx;
  thrust::device_vector<int> d_out(len);
#if defined(GATHER_TWO_ARRAYS)
  thrust::device_vector<int> d_out2(len);
#endif
  int tpb = 256;
  int blks = len / tpb / 2 + 1; // this mirrors thrust's grid configuration
  gather_kernel<<<blks, tpb>>>(
      d_idx.data().get(), (int)len,
      std::make_pair(d_out.data().get(), d_in.data().get())
#if defined(GATHER_TWO_ARRAYS)
          ,
      std::make_pair(d_out2.data().get(), d_in2.data().get())
#endif
  );
  // ignore warm up 1st run
  gather_kernel<<<blks, tpb>>>(
      d_idx.data().get(), (int)len,
      std::make_pair(d_out.data().get(), d_in.data().get())
#if defined(GATHER_TWO_ARRAYS)
          ,
      std::make_pair(d_out2.data().get(), d_in2.data().get())
#endif
  );
  thrust::host_vector<int> h_out = d_out;

  if (len < 20) {
    printf("custom gather_kernel\n");
    for (size_t i = 0; i < h_out.size(); ++i) {
      printf("%02d ", h_out[i]);
    }
    printf("\n");
  }

  // thrust gather
  thrust::fill(d_out.begin(), d_out.end(), 0); // reset
  thrust::gather(d_idx.begin(), d_idx.end(),   //
#if !defined(GATHER_TWO_ARRAYS)
                 d_in.begin(), d_out.begin());
#else
                 thrust::make_zip_iterator(d_in.begin(), d_in2.begin()),
                 thrust::make_zip_iterator(d_out.begin(), d_out2.begin()));
#endif
  // ignore warm up 1st run
  thrust::gather(d_idx.begin(), d_idx.end(), //
#if !defined(GATHER_TWO_ARRAYS)
                 d_in.begin(), d_out.begin());
#else
                 thrust::make_zip_iterator(d_in.begin(), d_in2.begin()),
                 thrust::make_zip_iterator(d_out.begin(), d_out2.begin()));
#endif
  h_out = d_out;
  if (len < 20) {
    printf("thrust::gather\n");
    for (size_t i = 0; i < h_out.size(); ++i) {
      printf("%02d ", h_out[i]);
    }
    printf("\n");
  }

  // thrust gather individually
#if defined(GATHER_TWO_ARRAYS)
  thrust::gather(d_idx.begin(), d_idx.end(), d_in.begin(), d_out.begin());
  thrust::gather(d_idx.begin(), d_idx.end(), d_in2.begin(), d_out2.begin());
  // repeat once
  thrust::gather(d_idx.begin(), d_idx.end(), d_in.begin(), d_out.begin());
  thrust::gather(d_idx.begin(), d_idx.end(), d_in2.begin(), d_out2.begin());
#endif

  return 0;
}

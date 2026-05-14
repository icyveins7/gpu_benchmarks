#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

struct CustomMin {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b < a) ? b : a;
  }
};

template <typename T> struct LessThanFunctor {
  T val;
  __device__ __forceinline__ bool operator()(const T &a) const {
    return a < val;
  }
};

TEST(ContainersCubw, DeviceSegmentedReduce) {
  using T = int;
  using Toffset = int;

  containers::DeviceImageStorage<T> d_in(32, 4);
  thrust::sequence(d_in.vec.begin(), d_in.vec.end());

  int numSegments = 2;
  thrust::device_vector<Toffset> d_begin_offsets{0, 64}; // 0th and 2nd row only
  thrust::device_vector<Toffset> d_end_offsets{32, 96};

  {
    thrust::device_vector<T> d_out(numSegments);
    cubw::DeviceSegmentedReduce::Reduce<T *, T *, Toffset *, Toffset *,
                                        CustomMin, T>
        cubw(numSegments);

    CustomMin min_op;
    cubw.exec(d_in.vec.data().get(), d_out.data().get(), numSegments,
              d_begin_offsets.data().get(), d_end_offsets.data().get(), min_op,
              1000);

    thrust::host_vector<T> h_out = d_out;
    ASSERT_EQ(h_out[0], 0);
    ASSERT_EQ(h_out[1], 64);
  }

  {
    thrust::device_vector<bool> d_out(numSegments);
    ::cuda::std::logical_or<bool> or_op;
    auto in_it = thrust::make_transform_iterator(d_in.vec.data().get(),
                                                 LessThanFunctor<T>{32});
    cubw::DeviceSegmentedReduce::Reduce<decltype(in_it), bool *, Toffset *,
                                        Toffset *,
                                        ::cuda::std::logical_or<bool>, bool>
        cubw(numSegments);

    cubw.exec(in_it, d_out.data().get(), numSegments,
              d_begin_offsets.data().get(), d_end_offsets.data().get(), or_op,
              false);

    thrust::host_vector<bool> h_out = d_out;
    // first row is less than, so it should be true
    ASSERT_EQ(h_out[0], true);
    // 3rd row is all more than, so it should be false
    ASSERT_EQ(h_out[1], false);
  }
}

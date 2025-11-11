#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/iterator/discard_iterator.h"
#include "thrust/reduce.h"
#include <cuda/std/functional> // for cuda::std::equal_to
#include <cuda_runtime.h>
#include <iostream>

template <typename Tkey, typename Tcoord, typename Tval> struct KeyCoordValue {
  Tkey key;
  Tcoord i;
  Tcoord j;
  Tval val;

  /**
   * @brief Used for transforming the iterator to obtain just the key.
   * NOTE: technically this may instantiate the values, just to perform a
   * conversion on another object; if this is not desired then move this to a
   * separate struct functor.
   *
   * @param kv Input
   * @return Key of the input
   */
  __host__ __device__ Tkey
  operator()(const KeyCoordValue<Tkey, Tcoord, Tval> &kv) {
    return kv.key;
  }
};

template <typename Tkey, typename Tcoord, typename Tval> struct KCVReduction {
  Tcoord i_min;
  Tcoord i_max;
  Tcoord j_min;
  Tcoord j_max;
  Tval minVal;
  Tval maxVal;

  __host__ __device__ KCVReduction
  operator()(const KeyCoordValue<Tkey, Tcoord, Tval> &kv) const {
    return KCVReduction{kv.i, kv.i, kv.j, kv.j, kv.val, kv.val};
  }
};

template <typename Tkey, typename Tcoord, typename Tval> struct KCVReducer {
  __host__ __device__ KCVReduction<Tkey, Tcoord, Tval>
  operator()(const KCVReduction<Tkey, Tcoord, Tval> &x,
             const KCVReduction<Tkey, Tcoord, Tval> &y) const {
    return KCVReduction<Tkey, Tcoord, Tval>{
        thrust::min(x.i_min, y.i_min),   thrust::max(x.i_max, y.i_max),
        thrust::min(x.j_min, y.j_min),   thrust::max(x.j_max, y.j_max),
        thrust::min(x.minVal, y.minVal), thrust::max(x.maxVal, y.maxVal)};
  }
};

int main() {
  printf("Thrust reduce examples\n");

  thrust::host_vector<KeyCoordValue<int, int, int>> h_x(10);
  h_x[0] = {0, 1, 2, 5};
  h_x[1] = {0, 3, 1, 1};

  h_x[2] = {1, 1, 4, 5};
  h_x[3] = {1, 9, 2, -6};

  h_x[4] = {2, 8, 7, 15};
  h_x[5] = {2, 6, 5, 25};
  h_x[6] = {2, 4, 3, -5};

  h_x[7] = {3, 8, 1, -15};
  h_x[8] = {3, 5, 3, 5};
  h_x[9] = {3, 3, 5, 3};
  thrust::device_vector<KeyCoordValue<int, int, int>> d_x(10);
  d_x = h_x;

  thrust::device_vector<KCVReduction<int, int, int>> d_y(10);
  thrust::reduce_by_key(
      thrust::make_transform_iterator(d_x.begin(),
                                      KeyCoordValue<int, int, int>()),
      thrust::make_transform_iterator(d_x.end(),
                                      KeyCoordValue<int, int, int>()),
      thrust::make_transform_iterator(
          d_x.begin(),
          KCVReduction<int, int, int>()), // turn the struct into the end result
      thrust::make_discard_iterator(),    // discard the 'keys'
      d_y.begin(),                        // output 'structs'
      cuda::std::equal_to<int>(),         // keys are just POD values
      KCVReducer<int, int, int>()         // custom functor to reduce
  );

  thrust::host_vector<KCVReduction<int, int, int>> h_y = d_y;

  for (size_t i = 0; i < h_y.size(); ++i) {
    printf("[%2zu] // i_min: %d, i_max: %d, j_min: %d, j_max: %d, minVal: %d, "
           "maxVal: %d\n",
           i, h_y[i].i_min, h_y[i].i_max, h_y[i].j_min, h_y[i].j_max,
           h_y[i].minVal, h_y[i].maxVal);
  }

  return 0;
}

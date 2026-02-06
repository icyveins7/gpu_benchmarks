#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T> struct SetNanToZero {
  __host__ __device__ T operator()(const T &x) const {
    return x == cuda::std::numeric_limits<T>::quiet_NaN() ? 0 : x;
  }
};

template <typename T> __global__ void readNanKernel(const T *x, int len) {
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    if (x[i] == cuda::std::numeric_limits<T>::quiet_NaN()) {
      printf("%d NAN via ==\n", i);
    }
    /*
    NOTE: BIG STUPID MISTAKE. DO NOT USE == (ABOVE) to CHECK FOR NANS. IT WILL
    FAIL.
    */
    if (std::isnan(x[i])) {
      printf("%d NAN via isnan\n", i);
    }
  }
}

int main() {

  thrust::host_vector<float> x{std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN(), 1, 2};
  thrust::device_vector<float> d_x = x;
  thrust::device_vector<char> d_tempstorage;
  size_t tempstorage;

  thrust::device_vector<float> d_y(1);

  auto inputIter =
      thrust::make_transform_iterator(d_x.begin(), SetNanToZero<float>{});

  cub::DeviceReduce::Sum(nullptr, tempstorage, inputIter, d_y.begin(),
                         d_x.size());
  d_tempstorage.resize(tempstorage);
  cub::DeviceReduce::Sum(d_tempstorage.data().get(), tempstorage, inputIter,
                         d_y.begin(), d_x.size());

  thrust::host_vector<float> y = d_y;
  printf("y = %f\n", y[0]);

  readNanKernel<<<1, 1>>>(d_x.data().get(), d_x.size());

  return 0;
}

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

// The following is adapted from
// cuda-samples/Samples/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu
struct Random_generator {

  int count;
  __host__ __device__ Random_generator() : count(0) {}
  __host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  __host__ __device__ __forceinline__ thrust::tuple<float, float> operator()() {

    // #ifdef __CUDA_ARCH__
    unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
    // thrust::generate may call operator() more than once per thread.
    // Hence, increment count by grid size to ensure uniqueness of seed
    count += blockDim.x * gridDim.x;

    // #else
    //
    //     unsigned seed = hash(0);
    //
    // #endif

    thrust::random::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib;

    return thrust::make_tuple(distrib(rng), distrib(rng));
  }
};

int main(int argc, char *argv[]) {
  printf("Thrust RNGs\n");

  size_t numPts = 1000000;
  if (argc >= 2)
    numPts = strtol(argv[1], nullptr, 10);
  printf("Using numPts = %ld\n", numPts);

  int repeats = 3;
  if (argc >= 3)
    repeats = strtol(argv[2], nullptr, 10);
  printf("Using repeats = %d\n", repeats);

  thrust::device_vector<float> d_x(numPts);
  thrust::device_vector<float> d_y(numPts);

  Random_generator rnd;
  for (int i = 0; i < repeats; ++i) {
    thrust::generate(
        thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end())),
        rnd);
  }
  thrust::host_vector<float> h_x = d_x;
  thrust::host_vector<float> h_y = d_y;

  for (int i = 0; i < 10; ++i) {
    printf("%.1f, %.1f\n", h_x[i], h_y[i]);
  }

  return 0;
}

/*
 * Here we define RNG-related structs or functions that use thrust.
 * These usually depend on vector-wide transform() or generate() functions
 * to implement the operator() function of the struct that is passed.
 */

#include <thrust/random.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

// The following is adapted from
// cuda-samples/Samples/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu
template <typename Engine = thrust::random::default_random_engine>
struct Random_generator2d {

  int count;
  __host__ __device__ Random_generator2d() : count(0) {}
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

    Engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib;

    return thrust::make_tuple(distrib(rng), distrib(rng));
  }
};

// Make a 1d flavour
struct Random_generator1d {
  int count;
  __host__ __device__ Random_generator1d() : count(0) {}
  __host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  __host__ __device__ __forceinline__ float operator()() {
    unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
    // thrust::generate may call operator() more than once per thread.
    // Hence, increment count by grid size to ensure uniqueness of seed
    count += blockDim.x * gridDim.x;
    thrust::random::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib;

    return distrib(rng);
  }
};

// 1D flavour that generates random phase to complex values
// i.e. exp(1i*rng)
struct Random_generator1d_phase {
  int count;
  __host__ __device__ Random_generator1d_phase() : count(0) {}
  __host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  __host__ __device__ __forceinline__ thrust::complex<float> operator()() {
    unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
    // thrust::generate may call operator() more than once per thread.
    // Hence, increment count by grid size to ensure uniqueness of seed
    count += blockDim.x * gridDim.x;
    thrust::random::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib(0, 2 * M_PI);

    float phase = distrib(rng);

    return thrust::complex<float>(cosf(phase), sinf(phase));
  }
};

// 1D phase to complex with a multiply
struct Random_generator1d_phase_cplxMul {
  int count;
  __host__ __device__ Random_generator1d_phase_cplxMul() : count(0) {}
  __host__ __device__ unsigned int hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
  }

  __host__ __device__ __forceinline__ thrust::complex<float> operator()(
    const thrust::complex<float>& input
  ) {
    unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
    // thrust::generate may call operator() more than once per thread.
    // Hence, increment count by grid size to ensure uniqueness of seed
    count += blockDim.x * gridDim.x;
    thrust::random::default_random_engine rng(seed);
    thrust::random::uniform_real_distribution<float> distrib(0, 2 * M_PI);

    float phase = distrib(rng);

    return thrust::complex<float>(cosf(phase), sinf(phase)) * input;
  }
};

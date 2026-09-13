// Slip-up: assuming a tensor-core (wmma) kernel on a second stream "just runs
// in parallel" with a CUDA-core kernel that saturates every SM.
//
// It does not. Running a kernel still requires thread blocks to be resident on
// an SM (register file / shared memory / warp slots). If kernel 1 fills every
// SM to its occupancy limit, kernel 2 has nowhere to be placed, so its stream
// blocks until kernel 1 finishes -- even though kernel 2 only touches the
// tensor cores.
//
// Proof: measure the wall-clock time of both kernels run together on two
// streams and compare it with the sum of their individual times.
//   overlap      ->  t_both ~= max(t_burn, t_wmma)
//   no overlap   ->  t_both ~= t_burn + t_wmma

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>

using namespace nvcuda;
using namespace nvcuda::wmma;

// Kernel 1: CUDA-core only. Occupies every SM to its occupancy limit and burns
// time in a scalar loop. `volatile` plus the global store stop the compiler
// from eliding the loop.
__global__ void burn(float* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  volatile float acc = 1.0f;
  for (int k = 0; k < iters; ++k) {
    acc = acc * 1.0000001f + 0.123456789f;
    acc = __fmaf_rn(3.0f, acc, 0.987654321f);
  }
  out[i] = acc;
}

// Kernel 2: tensor-core only. One warp per block computes one 16x16 C tile.
typedef wmma::fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
typedef wmma::fragment<matrix_b, 16, 16, 16, __half, col_major> frag_b;
typedef wmma::fragment<accumulator, 16, 16, 16, float> frag_c;

__global__ void wmma_kernel(const __half* A, const __half* B, float* C, int M,
                            int N, int K, int iters) {
  const int tilesPerRow = N / 16;
  const int mt = blockIdx.x / tilesPerRow;
  const int nt = blockIdx.x % tilesPerRow;

  frag_a af;
  frag_b bf;
  frag_c cf;
  wmma::fill_fragment(cf, 0.0f);

  wmma::load_matrix_sync(af, A + mt * 16 * K, K);
  wmma::load_matrix_sync(bf, B + nt * 16, N);

  for (int i = 0; i < iters; ++i) {
    wmma::mma_sync(cf, af, bf, cf);
  }

  wmma::store_matrix_sync(C + (mt * 16) * N + nt * 16, cf, N,
                          wmma::mem_row_major);
}

int main() {
  // ---- knobs (edit these to tune) ----
  const int burnBlockSize = 128;            // threads per burn block
  const int burnIters = 300000000;          // tune so t_burn ~= t_wmma
  const int wmmaIters = 150000000;          // tune so t_wmma ~= t_burn
  const int M = 64, N = 64, K = 64;         // 4x4 = 16 tiles of 16x16
  const int wmmaBlock = 32;                 // 1 warp per block
  const int wmmaGrid = (M / 16) * (N / 16); // 16 blocks

  int device = 0;
  cudaGetDevice(&device);

  int numSMs = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

  // Fill every SM to its occupancy limit so the wmma stream has nowhere to
  // place its blocks. Override maxBlocksPerSM by hand if you want to experiment
  // with partial occupancy.
  int maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, burn,
                                                burnBlockSize, 0);
  int burnGrid = maxBlocksPerSM * numSMs * 10; // * big number for good measure
  int burnTotalThreads = burnGrid * burnBlockSize;

  printf("device=%d  numSMs=%d  maxBlocksPerSM=%d  burnGrid=%d\n", device,
         numSMs, maxBlocksPerSM, burnGrid);

  __half* d_A = nullptr;
  __half* d_B = nullptr;
  float* d_C = nullptr;
  float* d_out = nullptr;
  cudaMalloc(&d_A, (size_t)M * K * sizeof(__half));
  cudaMalloc(&d_B, (size_t)K * N * sizeof(__half));
  cudaMalloc(&d_C, (size_t)M * N * sizeof(float));
  cudaMalloc(&d_out, (size_t)burnTotalThreads * sizeof(float));

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  cudaEvent_t eStart, eEnd;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eEnd);

  float t_burn = 0, t_wmma = 0, t_both = 0;

  // ---- kernel 1 alone ----
  cudaEventRecord(eStart, s1);
  burn<<<burnGrid, burnBlockSize, 0, s1>>>(d_out, burnTotalThreads, burnIters);
  cudaEventRecord(eEnd, s1);
  cudaStreamSynchronize(s1);
  cudaEventElapsedTime(&t_burn, eStart, eEnd);

  // ---- kernel 2 alone ----
  cudaEventRecord(eStart, s2);
  wmma_kernel<<<wmmaGrid, wmmaBlock, 0, s2>>>(d_A, d_B, d_C, M, N, K,
                                              wmmaIters);
  cudaEventRecord(eEnd, s2);
  cudaStreamSynchronize(s2);
  cudaEventElapsedTime(&t_wmma, eStart, eEnd);

  // ---- both together, one per stream ----
  cudaEventRecord(eStart, s1);
  burn<<<burnGrid, burnBlockSize, 0, s1>>>(d_out, burnTotalThreads, burnIters);
  wmma_kernel<<<wmmaGrid, wmmaBlock, 0, s2>>>(d_A, d_B, d_C, M, N, K,
                                              wmmaIters);
  cudaDeviceSynchronize();
  cudaEventRecord(eEnd, s1);
  cudaEventElapsedTime(&t_both, eStart, eEnd);

  printf("t_burn          = %.3f ms\n", t_burn);
  printf("t_wmma          = %.3f ms\n", t_wmma);
  printf("t_both(together) = %.3f ms\n", t_both);
  printf("t_burn + t_wmma     = %.3f ms\n", t_burn + t_wmma);

  // TODO: Not sure about this part yet
  // if (t_burn > 0.05f && t_wmma > 0.05f && t_both >= 0.95f * (t_burn +
  // t_wmma)) {
  //   printf(
  //       "RESULT: no overlap -> the wmma kernel waited for the burn
  //       kernel.\n");
  // } else if (t_both < 1.4f * ((t_burn > t_wmma) ? t_burn : t_wmma)) {
  //   printf("RESULT: overlap detected -> the kernels ran concurrently.\n");
  // } else {
  //   printf("RESULT: partial overlap; tune burnIters/wmmaIters so t_burn ~= "
  //          "t_wmma.\n");
  // }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_out);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaEventDestroy(eStart);
  cudaEventDestroy(eEnd);

  return 0;
}

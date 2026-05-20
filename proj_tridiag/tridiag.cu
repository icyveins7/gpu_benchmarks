#include <iostream>

#include "cpu_tridiag.h"
#include "tridiag.cuh"

#include "timer.h"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char *argv[]) {
  int NUM_ROWS = 2;
  int N = 64;
  int NUM_THREADS = 32;

  if (argc > 1)
    NUM_ROWS = atoi(argv[1]);
  if (argc > 2)
    N = atoi(argv[2]);
  if (argc > 3)
    NUM_THREADS = atoi(argv[3]);

  printf("Tridiagonal PCR test: %d rows x %d elements, %d threads/block\n",
         NUM_ROWS, N, NUM_THREADS);

  using namespace cutridiag;

  bool verbose = (NUM_ROWS <= 2 && N <= 64);

  srand(42);
  auto randf = []() { return (double)rand() / RAND_MAX; };

  // Generate random tridiagonal data, packed with stride N
  thrust::host_vector<double> ha(NUM_ROWS * N), hb(NUM_ROWS * N),
      hc(NUM_ROWS * N), hr(NUM_ROWS * N);

  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    ha[off] = 0.0;
    hb[off] = 4.0 + randf();
    hc[off] = randf();
    hr[off] = randf();
    for (int i = 1; i < N - 1; ++i) {
      ha[off + i] = randf();
      hb[off + i] = 4.0 + randf(); // diagonally dominant
      hc[off + i] = randf();
      hr[off + i] = randf();
    }
    ha[off + N - 1] = randf();
    hb[off + N - 1] = 4.0 + randf();
    hc[off + N - 1] = 0.0;
    hr[off + N - 1] = randf();
  }

  // CPU reference: solve each row independently
  thrust::host_vector<double> hu_cpu(NUM_ROWS * N), hgam(N);
  HighResolutionTimer<> timer;
  timer.start("cpuTridag");
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    tridag<double>(&ha[off], &hb[off], &hc[off], &hr[off], &hu_cpu[off],
                   hgam.data(), N);
  }
  timer.stop();

  // GPU PCR
  thrust::device_vector<double> da = ha, db = hb, dc = hc, dr = hr;
  thrust::device_vector<double> du(NUM_ROWS * N, 0.0);
  thrust::device_vector<size_t> dlen(NUM_ROWS, (size_t)N);
  TridiagPCRGlobalWorkspace<double> ws(NUM_ROWS, N);

  timer.start("gpuPCR");
  tridiag_blockwise_pcr_kernel<double><<<NUM_ROWS, NUM_THREADS>>>(
      da.data().get(), db.data().get(), dc.data().get(), dr.data().get(),
      du.data().get(), ws.buf0_a_ptr(), ws.buf0_b_ptr(), ws.buf0_c_ptr(),
      ws.buf0_rhs_ptr(), ws.buf1_a_ptr(), ws.buf1_b_ptr(), ws.buf1_c_ptr(),
      ws.buf1_rhs_ptr(), dlen.data().get(), N, NUM_ROWS);
  cudaDeviceSynchronize();
  timer.stop();

  thrust::host_vector<double> hu_gpu = du;

  // Compare
  double maxErr = 0.0;
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    if (verbose)
      printf("Row %d (n=%d, threads=%d):\n", row, N, NUM_THREADS);
    for (int i = 0; i < N; ++i) {
      auto err = fabs(hu_cpu[off + i] - hu_gpu[off + i]);
      if (err > maxErr)
        maxErr = err;
      if (verbose)
        printf("  [%2d] cpu=%+.10f  gpu=%+.10f  err=%e\n", i, hu_cpu[off + i],
               hu_gpu[off + i], err);
      if (err > 1e-10) {
        printf("LARGE ERROR at row %d, element %d\n", row, i);
        return 1;
      }
    }
    if (verbose)
      printf("-------------\n");
  }
  printf("Max error: %e\n", maxErr);

  return 0;
}

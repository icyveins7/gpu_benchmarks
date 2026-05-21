#include <iostream>

#include "cpu_tridiag.h"
#include "tridiag.cuh"

#include "timer.h"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef USE_FLOAT
using Real = float;
constexpr double ERR_TOL = 1e-5;
constexpr const char *TYPE_STR = "float";
#else
using Real = double;
constexpr double ERR_TOL = 1e-10;
constexpr const char *TYPE_STR = "double";
#endif

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

  printf("Tridiagonal PCR test (%s): %d rows x %d elements, %d threads/block\n",
         TYPE_STR, NUM_ROWS, N, NUM_THREADS);

  using namespace cutridiag;

  bool verbose = (NUM_ROWS <= 2 && N <= 64);

  srand(42);
  auto randf = []() { return (Real)rand() / RAND_MAX; };

  // Generate random tridiagonal data, packed with stride N
  thrust::host_vector<Real> ha(NUM_ROWS * N), hb(NUM_ROWS * N),
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
  thrust::host_vector<Real> hu_cpu(NUM_ROWS * N), hgam(N);
  HighResolutionTimer<> timer;
  timer.start("cpuTridag");
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    tridag<Real>(&ha[off], &hb[off], &hc[off], &hr[off], &hu_cpu[off],
                 hgam.data(), N);
  }
  timer.stop();

  // GPU PCR
  thrust::device_vector<Real> da = ha, db = hb, dc = hc, dr = hr;
  thrust::device_vector<Real> du(NUM_ROWS * N, Real(0));
  thrust::device_vector<size_t> dlen(NUM_ROWS, (size_t)N);
  TridiagPCRWorkspace<Real> ws(NUM_ROWS, N);

  timer.start("gpuPCR");
  tridiag_blockwise_pcr_kernel<Real><<<NUM_ROWS, NUM_THREADS>>>(
      da.data().get(), db.data().get(), dc.data().get(), dr.data().get(),
      du.data().get(), ws.buf0_a_ptr(), ws.buf0_b_ptr(), ws.buf0_c_ptr(),
      ws.buf0_rhs_ptr(), ws.buf1_a_ptr(), ws.buf1_b_ptr(), ws.buf1_c_ptr(),
      ws.buf1_rhs_ptr(), dlen.data().get(), N, NUM_ROWS);
  cudaDeviceSynchronize();
  timer.stop();

  thrust::host_vector<Real> hu_gpu = du;

  // Compare
  Real maxErr = 0.0;
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
      if (err > ERR_TOL) {
        printf("LARGE ERROR at row %d, element %d\n", row, i);
        return 1;
      }
    }
    if (verbose)
      printf("-------------\n");
  }
  printf("Max error (global mem): %e\n", maxErr);

  // GPU PCR with shared memory
  {
    // Determine the shmem workspace requirement, same struct but static method
    size_t shmemBytes = TridiagPCRWorkspace<Real>::requiredShmemBytes(N);

    int device;
    cudaGetDevice(&device);
    int maxShmem;
    cudaDeviceGetAttribute(&maxShmem, cudaDevAttrMaxSharedMemoryPerBlock,
                           device);

    if (shmemBytes > (size_t)maxShmem) {
      printf("Shmem PCR skipped: need %zu bytes, device limit is %d bytes\n",
             shmemBytes, maxShmem);
    } else {
      printf("Shmem PCR: %zu bytes requested (device max %d bytes)\n",
             shmemBytes, maxShmem);

      thrust::fill(du.begin(), du.end(), Real(0));

      timer.start("gpuPCR_shmem");
      tridiag_blockwise_pcr_shmem_kernel<Real>
          <<<NUM_ROWS, NUM_THREADS, shmemBytes>>>(
              da.data().get(), db.data().get(), dc.data().get(),
              dr.data().get(), du.data().get(), dlen.data().get(), N, NUM_ROWS);
      cudaDeviceSynchronize();
      timer.stop();

      hu_gpu = du;

      Real maxErrShmem = 0.0;
      for (int row = 0; row < NUM_ROWS; ++row) {
        int off = row * N;
        if (verbose)
          printf("Shmem Row %d (n=%d, threads=%d):\n", row, N, NUM_THREADS);
        for (int i = 0; i < N; ++i) {
          auto err = fabs(hu_cpu[off + i] - hu_gpu[off + i]);
          if (err > maxErrShmem)
            maxErrShmem = err;
          if (verbose)
            printf("  [%2d] cpu=%+.10f  gpu=%+.10f  err=%e\n", i,
                   hu_cpu[off + i], hu_gpu[off + i], err);
          if (err > ERR_TOL) {
            printf("LARGE ERROR (shmem) at row %d, element %d\n", row, i);
            return 1;
          }
        }
        if (verbose)
          printf("-------------\n");
      }
      printf("Max error (shmem): %e\n", maxErrShmem);
    }
  }

  // ============================================
  // ============================================
  // ============== CYCLIC ======================
  // ============================================
  // ============================================

  printf("\n--- Cyclic tridiagonal test ---\n");

  // Generate cyclic tridiagonal data (all rows same length N)
  // a[0] = alpha (bottom-left corner), c[N-1] = beta (top-right corner)
  thrust::host_vector<Real> cha(NUM_ROWS * N), chb(NUM_ROWS * N),
      chc(NUM_ROWS * N), chr(NUM_ROWS * N);

  srand(123);
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    for (int i = 0; i < N; ++i) {
      cha[off + i] = randf();
      chb[off + i] = Real(4) + randf(); // diagonally dominant
      chc[off + i] = randf();
      chr[off + i] = randf();
    }
    // a[0] and c[N-1] are the cyclic corner elements (kept in-place)
  }

  // CPU reference: cyclic solver
  // cyclic() expects a[0]=0, c[N-1]=0 with corners passed separately,
  // so we extract them and zero the positions for the CPU call
  thrust::host_vector<Real> cha_cpu = cha, chc_cpu = chc;
  thrust::host_vector<Real> chu_cpu(NUM_ROWS * N);
  thrust::host_vector<Real> cbb(N), cu(N), cz(N), cgam(N);
  timer.start("cpuCyclic");
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    Real beta = cha_cpu[off];          // top-right (row 0's unused a)
    Real alpha = chc_cpu[off + N - 1]; // bottom-left (row N-1's unused c)
    cha_cpu[off] = Real(0);
    chc_cpu[off + N - 1] = Real(0);
    cyclic<Real>(&cha_cpu[off], &chb[off], &chc_cpu[off], alpha, beta,
                 &chr[off], &chu_cpu[off], cbb.data(), cu.data(), cz.data(),
                 cgam.data(), N);
  }
  timer.stop();

  // GPU cyclic PCR (corners read directly from a[0] and c[N-1])
  thrust::device_vector<Real> cda = cha, cdb = chb, cdc = chc, cdr = chr;
  thrust::device_vector<Real> cdu(NUM_ROWS * N, Real(0));
  thrust::device_vector<size_t> cdlen(NUM_ROWS, (size_t)N);
  TridiagPCRWorkspace<Real> cws(NUM_ROWS, N);

  timer.start("gpuCyclicPCR");
  cyclic_tridiag_blockwise_pcr_kernel<Real><<<NUM_ROWS, NUM_THREADS>>>(
      cda.data().get(), cdb.data().get(), cdc.data().get(), cdr.data().get(),
      cdu.data().get(), cws.buf0_a_ptr(), cws.buf0_b_ptr(), cws.buf0_c_ptr(),
      cws.buf0_rhs_ptr(), cws.buf1_a_ptr(), cws.buf1_b_ptr(),
      cws.buf1_c_ptr(), cws.buf1_rhs_ptr(), cdlen.data().get(), N, NUM_ROWS);
  cudaDeviceSynchronize();
  timer.stop();

  thrust::host_vector<Real> chu_gpu = cdu;

  // Compare
  Real maxErrCyclic = 0.0;
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    if (verbose)
      printf("Cyclic Row %d (n=%d, threads=%d):\n", row, N, NUM_THREADS);
    for (int i = 0; i < N; ++i) {
      auto err = fabs(chu_cpu[off + i] - chu_gpu[off + i]);
      if (err > maxErrCyclic)
        maxErrCyclic = err;
      if (verbose)
        printf("  [%2d] cpu=%+.10f  gpu=%+.10f  err=%e\n", i,
               chu_cpu[off + i], chu_gpu[off + i], err);
      if (err > ERR_TOL) {
        printf("LARGE ERROR (cyclic) at row %d, element %d\n", row, i);
        return 1;
      }
    }
    if (verbose)
      printf("-------------\n");
  }
  printf("Max error (cyclic global mem): %e\n", maxErrCyclic);

  return 0;
}

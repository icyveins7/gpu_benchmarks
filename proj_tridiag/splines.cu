#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "splines.cuh"

#include "timer.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef USE_FLOAT
using Real = float;
constexpr const char *TYPE_STR = "float";
#else
using Real = double;
constexpr const char *TYPE_STR = "double";
#endif

int main(int argc, char *argv[]) {
  int NUM_ROWS = 2;
  int N = 10; // points per row
  int NUM_THREADS = 32;

  if (argc > 1)
    NUM_ROWS = atoi(argv[1]);
  if (argc > 2)
    N = atoi(argv[2]);
  if (argc > 3)
    NUM_THREADS = atoi(argv[3]);

  printf("Natural spline test (%s): %d rows x %d points, %d threads/block\n",
         TYPE_STR, NUM_ROWS, N, NUM_THREADS);

  using namespace cutridiag;
  using namespace cuspline;

  bool verbose = (NUM_ROWS <= 2 && N <= 20);

  // Generate sample data: x uniform, y = sin(x) + noise
  srand(42);
  auto randf = []() { return Real(0.1) * (Real)rand() / RAND_MAX; };

  thrust::host_vector<Real> hx(NUM_ROWS * N), hy(NUM_ROWS * N);

  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    for (int i = 0; i < N; ++i) {
      hx[off + i] = Real(i); // uniform spacing
      hy[off + i] = sin(Real(i) * Real(0.5)) + randf();
    }
  }

  if (verbose) {
    for (int row = 0; row < NUM_ROWS; ++row) {
      int off = row * N;
      printf("Row %d data:\n  x = {", row);
      for (int i = 0; i < N; ++i)
        printf("%s%.8f", i ? ", " : "", hx[off + i]);
      printf("}\n  y = {");
      for (int i = 0; i < N; ++i)
        printf("%s%.8f", i ? ", " : "", hy[off + i]);
      printf("}\n");
    }
  }

  // Upload to device
  thrust::device_vector<Real> dx = hx, dy = hy;
  thrust::device_vector<size_t> dlen(NUM_ROWS, (size_t)N);

  // Allocate spline output: N-1 intervals per row, stride N
  thrust::device_vector<CubicSpline<Real>> dsplines(NUM_ROWS * N);

  // PCR workspace
  TridiagPCRWorkspace<Real> ws(NUM_ROWS, N);

  HighResolutionTimer<> timer;
  timer.start("gpuNaturalSpline");
  natural_spline_blockwise_kernel<Real><<<NUM_ROWS, NUM_THREADS>>>(
      dx.data().get(), dy.data().get(), dsplines.data().get(), ws.buf0_a_ptr(),
      ws.buf0_b_ptr(), ws.buf0_c_ptr(), ws.buf0_rhs_ptr(), ws.buf1_a_ptr(),
      ws.buf1_b_ptr(), ws.buf1_c_ptr(), ws.buf1_rhs_ptr(), dlen.data().get(), N,
      NUM_ROWS);
  cudaDeviceSynchronize();
  timer.stop();

  // Copy back
  thrust::host_vector<CubicSpline<Real>> hsplines = dsplines;

  // Print results
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    if (verbose)
      printf("Row %d:\n", row);
    for (int i = 0; i < N - 1; ++i) {
      const auto &s = hsplines[off + i];
      if (verbose) {
        printf("  [%2d] x=[%+.4f, %+.4f]  a=%+.6f  b=%+.6f  c=%+.6f  d=%+.6f\n",
               i, s.xmin, s.xmax, s.a, s.b, s.c, s.d);

        // Verify continuity: S_i(xmax) should equal y[i+1]
        Real yEnd = s.evaluate(s.xmax);
        printf("       S(%+.4f) = %+.10f  y[%d] = %+.10f  diff = %e\n", s.xmax,
               yEnd, i + 1, hy[off + i + 1], fabs(yEnd - hy[off + i + 1]));
      }
    }
    if (verbose)
      printf("-------------\n");
  }

  // ============================================
  // ============== CLAMPED =====================
  // ============================================

  printf("\n--- Clamped spline test ---\n");

  // Per-row clamped slopes
  thrust::host_vector<Real> hSlopeLeft(NUM_ROWS), hSlopeRight(NUM_ROWS);
  for (int row = 0; row < NUM_ROWS; ++row) {
    hSlopeLeft[row] = Real(0.5);   // prescribed slope at x[0]
    hSlopeRight[row] = Real(-0.3); // prescribed slope at x[N-1]
  }
  thrust::device_vector<Real> dSlopeLeft = hSlopeLeft;
  thrust::device_vector<Real> dSlopeRight = hSlopeRight;

  thrust::device_vector<CubicSpline<Real>> dsplines_clamped(NUM_ROWS * N);

  // Clamped needs N equations, so workspace must fit N (already does since ws
  // was sized for N)
  timer.start("gpuClampedSpline");
  clamped_spline_blockwise_kernel<Real><<<NUM_ROWS, NUM_THREADS>>>(
      dx.data().get(), dy.data().get(), dsplines_clamped.data().get(),
      ws.buf0_a_ptr(), ws.buf0_b_ptr(), ws.buf0_c_ptr(), ws.buf0_rhs_ptr(),
      ws.buf1_a_ptr(), ws.buf1_b_ptr(), ws.buf1_c_ptr(), ws.buf1_rhs_ptr(),
      dlen.data().get(), N, NUM_ROWS, dSlopeLeft.data().get(),
      dSlopeRight.data().get());
  cudaDeviceSynchronize();
  timer.stop();

  thrust::host_vector<CubicSpline<Real>> hsplines_clamped = dsplines_clamped;

  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    if (verbose)
      printf("Clamped Row %d (slopeLeft=%.4f, slopeRight=%.4f):\n", row,
             hSlopeLeft[row], hSlopeRight[row]);
    for (int i = 0; i < N - 1; ++i) {
      const auto &s = hsplines_clamped[off + i];
      if (verbose) {
        printf("  [%2d] x=[%+.4f, %+.4f]  a=%+.6f  b=%+.6f  c=%+.6f  d=%+.6f\n",
               i, s.xmin, s.xmax, s.a, s.b, s.c, s.d);

        Real yEnd = s.evaluate(s.xmax);
        printf("       S(%+.4f) = %+.10f  y[%d] = %+.10f  diff = %e\n", s.xmax,
               yEnd, i + 1, hy[off + i + 1], fabs(yEnd - hy[off + i + 1]));
      }
    }
    if (verbose)
      printf("-------------\n");
  }

  // ============================================
  // ============== PERIODIC ====================
  // ============================================

  printf("\n--- Periodic spline test ---\n");

  // For periodic splines, y must wrap: y(x[0] + xPeriod) == y(x[0]).
  // Generate periodic data: y = sin(2*pi*x / xPeriod)
  // Per-row periods
  thrust::host_vector<Real> hXPeriod(NUM_ROWS);
  for (int row = 0; row < NUM_ROWS; ++row) {
    hXPeriod[row] = Real(N); // period = N so x goes from 0..N-1
  }
  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    for (int i = 0; i < N; ++i) {
      hx[off + i] = Real(i);
      hy[off + i] = sin(Real(2) * Real(M_PI) * Real(i) / Real(N)) + randf();
    }
  }

  if (verbose) {
    printf("Periodic data (xPeriod=%.4f):\n", hXPeriod[0]);
    for (int row = 0; row < NUM_ROWS; ++row) {
      int off = row * N;
      printf("Row %d data:\n  x = {", row);
      for (int i = 0; i < N; ++i)
        printf("%s%.8f", i ? ", " : "", hx[off + i]);
      printf("}\n  y = {");
      for (int i = 0; i < N; ++i)
        printf("%s%.8f", i ? ", " : "", hy[off + i]);
      printf("}\n");
    }
  }

  // Re-upload modified data
  dx = hx;
  dy = hy;

  // Periodic produces N intervals per row (including wrap-around)
  thrust::device_vector<CubicSpline<Real>> dsplines_periodic(NUM_ROWS * N);

  // Cyclic solver needs separate input arrays (survive across 2 PCR solves)
  TridiagPCRWorkspace<Real> ws_arr(NUM_ROWS, N);    // arranged coefficients
  thrust::device_vector<Real> dz_out(NUM_ROWS * N); // z output
  thrust::device_vector<Real> dXPeriod = hXPeriod;

  timer.start("gpuPeriodicSpline");
  periodic_spline_blockwise_kernel<Real><<<NUM_ROWS, NUM_THREADS>>>(
      dx.data().get(), dy.data().get(), dsplines_periodic.data().get(),
      ws_arr.buf0_a_ptr(), ws_arr.buf0_b_ptr(), ws_arr.buf0_c_ptr(),
      ws_arr.buf0_rhs_ptr(), ws.buf0_a_ptr(), ws.buf0_b_ptr(), ws.buf0_c_ptr(),
      ws.buf0_rhs_ptr(), ws.buf1_a_ptr(), ws.buf1_b_ptr(), ws.buf1_c_ptr(),
      ws.buf1_rhs_ptr(), dz_out.data().get(), dlen.data().get(), N, NUM_ROWS,
      dXPeriod.data().get());
  cudaDeviceSynchronize();
  timer.stop();

  thrust::host_vector<CubicSpline<Real>> hsplines_periodic = dsplines_periodic;

  for (int row = 0; row < NUM_ROWS; ++row) {
    int off = row * N;
    if (verbose)
      printf("Periodic Row %d:\n", row);
    for (int i = 0; i < N; ++i) {
      const auto &s = hsplines_periodic[off + i];
      if (verbose) {
        printf("  [%2d] x=[%+.4f, %+.4f]  a=%+.6f  b=%+.6f  c=%+.6f  d=%+.6f\n",
               i, s.xmin, s.xmax, s.a, s.b, s.c, s.d);

        // Continuity: S_i(xmax) should equal y[i+1] (wrapping for last)
        Real yEnd = s.evaluate(s.xmax);
        int inext = (i + 1) % N;
        printf("       S(%+.4f) = %+.10f  y[%d] = %+.10f  diff = %e\n", s.xmax,
               yEnd, inext, hy[off + inext], fabs(yEnd - hy[off + inext]));
      }
    }
    if (verbose)
      printf("-------------\n");
  }

  return 0;
}

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
        printf("%s%.4f", i ? ", " : "", hx[off + i]);
      printf("}\n  y = {");
      for (int i = 0; i < N; ++i)
        printf("%s%.4f", i ? ", " : "", hy[off + i]);
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

  return 0;
}

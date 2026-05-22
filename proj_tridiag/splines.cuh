#pragma once

#include "tridiag.cuh"

#include "extra_type_traits.cuh"

#include <cstdint>

namespace cuspline {

/**
 * @brief Per-interval cubic spline representation.
 * S(x) = a + b*t + c*t^2 + d*t^3,  where t = x - xmin.
 *
 * Coefficients from second derivatives z_i:
 *   a = y_i
 *   b = (y_{i+1} - y_i)/h - h*z_{i+1}/6 - h*z_i/3
 *   c = z_i / 2
 *   d = (z_{i+1} - z_i) / (6*h)
 * where h = xmax - xmin.
 */
template <typename T> struct CubicSpline {
  T xmin, xmax;
  T a, b, c, d;

  __host__ __device__ T evaluate(T xPt) const {
    T t = xPt - xmin;
    return a + t * (b + t * (c + t * d));
  }

  __host__ __device__ T evaluateSlope(T xPt) const {
    T t = xPt - xmin;
    return b + t * (T(2) * c + t * T(3) * d);
  }
};

/**
 * @brief Compute one CubicSpline's coefficients from the tridiag solution.
 * @param s      Output spline for this interval
 * @param xi     Left x-coordinate
 * @param h      Interval width (x_{i+1} - x_i)
 * @param yi     Left y-value
 * @param yi1    Right y-value
 * @param zi     Second derivative at left knot
 * @param zi1    Second derivative at right knot
 */
template <typename T>
__device__ void set_spline_coeffs_from_tridiag_solution(CubicSpline<T> &s, T xi,
                                                        T h, T yi, T yi1, T zi,
                                                        T zi1) {
  s.xmin = xi;
  s.xmax = xi + h;
  s.a = yi;
  s.b = (yi1 - yi) / h - h * zi1 / T(6) - h * zi / T(3);
  s.c = zi / T(2);
  s.d = (zi1 - zi) / (T(6) * h);
}

/**
 * @brief Computes the width of the i'th interval for an array of x-coordinates.
 *
 * @detail The i'th interval is defined as the distance from x[i] to x[i+1].
 * As such there are N-1 intervals for N points.
 *
 * @tparam T Type of x-coordinates
 * @param x Array of x-coordinates
 * @param length Number of x-coordinates
 * @param intervalIdx Requested interval index
 * @return Width of the i'th interval, 0 for invalid intervals
 */
template <typename T>
__device__ __forceinline__ T xWidth(const T *x, const int length,
                                    const int intervalIdx) {
  // Interval index i is defined from x[i] to x[i+1],
  // valid range is 0 <= i <= length-2 (i.e. N-1 intervals for N points)
  if (intervalIdx < 0 || intervalIdx >= length - 1) {
    return T(0);
  } else {
    return x[intervalIdx + 1] - x[intervalIdx];
  }
}

/**
 * @brief Natural spline: N points -> N-2 equations for z_1 .. z_{N-2}.
 * z_0 = z_{N-1} = 0 are not part of the system.
 */
template <typename T>
__device__ void
arrange_natural_spline_blockwide(const T *x, const T *y, const int length,
                                 cutridiag::tridiag_pcr_scratch<T> &params) {
  // Total N-2 equations for N points
  for (int t = threadIdx.x; t < length - 2; t += blockDim.x) {
    // first equation starts from j = 1 i.e. first interior knot
    int j = t + 1;
    // interval on the right is index j
    T rightWidth = xWidth(x, length, j);
    // interval on the left is index j - 1
    T leftWidth = xWidth(x, length, j - 1);

    params.buf0_a[t] = leftWidth;
    if (j - 1 <= 0) {
      params.buf0_a[t] = 0; // first equation has a = 0
    }

    params.buf0_b[t] = T(2) * (leftWidth + rightWidth);

    params.buf0_c[t] = rightWidth;
    if (j + 1 >= length - 1) {
      params.buf0_c[t] = 0; // last equation has c = 0
    }

    params.buf0_rhs[t] =
        T(6) * ((y[j + 1] - y[j]) / rightWidth - (y[j] - y[j - 1]) / leftWidth);
  }
}

/**
 * @brief Clamped spline: N points -> N equations for z_0 .. z_{N-1}.
 * Prescribed slopes at both endpoints.
 */
template <typename T>
__device__ void
arrange_clamped_spline_blockwide(const T *x, const T *y, const int length,
                                 cutridiag::tridiag_pcr_scratch<T> &params,
                                 T slopeLeft, T slopeRight) {
  // Total N equations for N points, unknowns z_0 .. z_{N-1}
  for (int t = threadIdx.x; t < length; t += blockDim.x) {
    if (t == 0) {
      // Row 0: (h0/3)*z0 + (h0/6)*z1 = (y1-y0)/h0 - slopeLeft
      T h0 = xWidth(x, length, 0);
      params.buf0_a[0] = T(0);
      params.buf0_b[0] = h0 / T(3);
      params.buf0_c[0] = h0 / T(6);
      params.buf0_rhs[0] = (y[1] - y[0]) / h0 - slopeLeft;
    } else if (t == length - 1) {
      // Row N-1: (h_{N-2}/6)*z_{N-2} + (h_{N-2}/3)*z_{N-1} = slopeRight -
      // (y_{N-1}-y_{N-2})/h_{N-2}
      T hLast = xWidth(x, length, length - 2);
      params.buf0_a[length - 1] = hLast / T(6);
      params.buf0_b[length - 1] = hLast / T(3);
      params.buf0_c[length - 1] = T(0);
      params.buf0_rhs[length - 1] =
          slopeRight - (y[length - 1] - y[length - 2]) / hLast;
    } else {
      // Interior row t: h_{t-1}*z_{t-1} + 2(h_{t-1}+h_t)*z_t + h_t*z_{t+1} =
      // rhs
      T leftWidth = xWidth(x, length, t - 1);
      T rightWidth = xWidth(x, length, t);
      params.buf0_a[t] = leftWidth;
      params.buf0_b[t] = T(2) * (leftWidth + rightWidth);
      params.buf0_c[t] = rightWidth;
      params.buf0_rhs[t] = T(6) * ((y[t + 1] - y[t]) / rightWidth -
                                   (y[t] - y[t - 1]) / leftWidth);
    }
  }
}

/**
 * @brief Periodic spline: N real points -> N equations for z_0 .. z_{N-1}.
 * The caller supplies N distinct points (x_0,y_0)..(x_{N-1},y_{N-1}) and the
 * period xPeriod. The function value wraps: y(x_0 + xPeriod) = y(x_0), so
 * the caller does NOT need to duplicate the first point at the end.
 *
 * @example N=4 real points, xPeriod = x_0 + P:
 *
 *     stored in x[]:                not stored
 *      |                              |
 *     x_0     x_1       x_2   x_3  (x_0 + P)
 *      *-------*---------*-----*-----(*)
 *      |  h_0  |   h_1   | h_2 | h_3  |
 *      |                              |
 *      |<---------- xPeriod --------->|
 *
 *     (*) = synthetic periodic point, y = y_0
 *           h_3 = xPeriod - (x_3 - x_0)
 *
 * Produces an N x N cyclic tridiagonal system with corners in a[0] and c[N-1].
 * Use cyclic_tridiag_blockwide_pcr to solve.
 *
 * @param x        x-coordinates, length N (no duplicated endpoint)
 * @param y        y-values, length N
 * @param length   N, the number of real data points
 * @param xPeriod  the period: x_{N} would be x_0 + xPeriod
 * @param out_a    Output sub-diagonal coefficients (length N)
 * @param out_b    Output diagonal coefficients (length N)
 * @param out_c    Output super-diagonal coefficients (length N)
 * @param out_rhs  Output RHS coefficients (length N)
 *
 * Output arrays are separate from PCR workspace because the cyclic solver
 * needs to reuse the coefficients across two PCR solves.
 */
template <typename T>
__device__ void
arrange_periodic_spline_blockwide(const T *x, const T *y, const int length,
                                  const T xPeriod, T *out_a, T *out_b, T *out_c,
                                  T *out_rhs) {
  // N real points, N intervals (last one wraps), N unknowns z_0 .. z_{N-1}
  // with z_N = z_0 by periodicity
  int N = length;

  // Last interval width: from x[N-1] to x[0]+xPeriod
  T hLast = xPeriod - (x[N - 1] - x[0]);

  for (int t = threadIdx.x; t < N; t += blockDim.x) {
    // h_t = x[t+1] - x[t], with h_{N-1} = hLast
    T rightWidth = (t < N - 1) ? xWidth(x, N, t) : hLast;
    // h_{t-1}, with h_{-1} = hLast
    T leftWidth = (t > 0) ? xWidth(x, N, t - 1) : hLast;

    out_a[t] = leftWidth;
    out_b[t] = T(2) * (leftWidth + rightWidth);
    out_c[t] = rightWidth;

    // y neighbours wrap: y_{-1} = y_{N-1}, y_{N} = y_0
    T yLeft = (t > 0) ? y[t - 1] : y[N - 1];
    T yRight = (t < N - 1) ? y[t + 1] : y[0];

    out_rhs[t] =
        T(6) * ((yRight - y[t]) / rightWidth - (y[t] - yLeft) / leftWidth);
  }
}

/**
 * @brief Kernel: fit natural cubic splines for multiple rows.
 * One CUDA block per row. Each row has num_points_in_row[blk] data points,
 * producing N-1 CubicSpline<T> intervals per row.
 *
 * The buf0 arrays serve double duty: arrange writes the tridiag coefficients
 * into them, then the PCR solver uses them as the initial buffer.
 *
 * @param x       Input x-coordinates, strided (row blk starts at blk*stride)
 * @param y       Input y-values, same layout
 * @param splines Output CubicSpline array, N-1 per row (same stride)
 * @param buf0_a    PCR workspace buffer 0, sub-diagonal
 * @param buf0_b    PCR workspace buffer 0, diagonal
 * @param buf0_c    PCR workspace buffer 0, super-diagonal
 * @param buf0_rhs  PCR workspace buffer 0, RHS
 * @param buf1_a    PCR workspace buffer 1, sub-diagonal
 * @param buf1_b    PCR workspace buffer 1, diagonal
 * @param buf1_c    PCR workspace buffer 1, super-diagonal
 * @param buf1_rhs  PCR workspace buffer 1, RHS
 * @param num_points_in_row  Number of data points N per row
 * @param stride_elements_per_row  Stride between rows in all arrays
 * @param num_rows  Total number of rows
 */
template <typename T>
__global__ void natural_spline_blockwise_kernel(
    const T *x, const T *y, CubicSpline<T> *splines, T *buf0_a, T *buf0_b,
    T *buf0_c, T *buf0_rhs, T *buf1_a, T *buf1_b, T *buf1_c, T *buf1_rhs,
    const size_t *num_points_in_row, const int stride_elements_per_row,
    const int num_rows) {

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    int N = num_points_in_row[blk];
    int M = N - 2; // number of equations

    cutridiag::tridiag_pcr_scratch<T> params = {
        &buf0_a[off],   &buf0_b[off],   &buf0_c[off],
        &buf0_rhs[off], &buf1_a[off],   &buf1_b[off],
        &buf1_c[off],   &buf1_rhs[off], (size_t)M};

    // Arrange writes N-2 equations directly into buf0
    arrange_natural_spline_blockwide(&x[off], &y[off], N, params);
    __syncthreads();

    // Solve: buf0 is already populated, PCR reduces in-place
    T *rFinal, *bFinal;
    cutridiag::pcr_reduce_blockwide<T>(
        params.buf0_a, params.buf0_b, params.buf0_c, params.buf0_rhs,
        params.buf1_a, params.buf1_b, params.buf1_c, params.buf1_rhs, M, rFinal,
        bFinal);

    // Natural BC: z_0 = z_{N-1} = 0. The solver only produces M = N-2
    // interior values z_1..z_{N-2}, stored in rFinal[0..M-1].
    // So for interval i, zi maps to rFinal[i-1] and zi1 to rFinal[i],
    // with boundary cases returning 0.
    for (int i = threadIdx.x; i < N - 1; i += blockDim.x) {
      T zi = (i > 0 && i <= M) ? rFinal[i - 1] / bFinal[i - 1] : T(0);
      T zi1 = (i + 1 > 0 && i + 1 <= M) ? rFinal[i] / bFinal[i] : T(0);
      set_spline_coeffs_from_tridiag_solution(splines[off + i], x[off + i],
                                              xWidth(&x[off], N, i), y[off + i],
                                              y[off + i + 1], zi, zi1);
    }
    __syncthreads();
  }
}

/**
 * @brief Kernel: fit clamped cubic splines for multiple rows.
 * One CUDA block per row. Each row has num_points_in_row[blk] data points,
 * producing N-1 CubicSpline<T> intervals per row.
 * N equations for N unknowns z_0..z_{N-1}.
 *
 * @param x       Input x-coordinates, strided (row blk starts at blk*stride)
 * @param y       Input y-values, same layout
 * @param splines Output CubicSpline array, N-1 per row (same stride)
 * @param buf0_a    PCR workspace buffer 0, sub-diagonal
 * @param buf0_b    PCR workspace buffer 0, diagonal
 * @param buf0_c    PCR workspace buffer 0, super-diagonal
 * @param buf0_rhs  PCR workspace buffer 0, RHS
 * @param buf1_a    PCR workspace buffer 1, sub-diagonal
 * @param buf1_b    PCR workspace buffer 1, diagonal
 * @param buf1_c    PCR workspace buffer 1, super-diagonal
 * @param buf1_rhs  PCR workspace buffer 1, RHS
 * @param num_points_in_row  Number of data points N per row
 * @param stride_elements_per_row  Stride between rows in all arrays
 * @param num_rows  Total number of rows
 * @param slopeLeft   Prescribed slope at the left endpoint
 * @param slopeRight  Prescribed slope at the right endpoint
 */
template <typename T>
__global__ void clamped_spline_blockwise_kernel(
    const T *x, const T *y, CubicSpline<T> *splines, T *buf0_a, T *buf0_b,
    T *buf0_c, T *buf0_rhs, T *buf1_a, T *buf1_b, T *buf1_c, T *buf1_rhs,
    const size_t *num_points_in_row, const int stride_elements_per_row,
    const int num_rows, T slopeLeft, T slopeRight) {

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    int N = num_points_in_row[blk];
    int M = N; // number of equations

    cutridiag::tridiag_pcr_scratch<T> params = {
        &buf0_a[off],   &buf0_b[off],   &buf0_c[off],
        &buf0_rhs[off], &buf1_a[off],   &buf1_b[off],
        &buf1_c[off],   &buf1_rhs[off], (size_t)M};

    // Arrange writes N equations directly into buf0
    arrange_clamped_spline_blockwide(&x[off], &y[off], N, params, slopeLeft,
                                     slopeRight);
    __syncthreads();

    // Solve
    T *rFinal, *bFinal;
    cutridiag::pcr_reduce_blockwide<T>(
        params.buf0_a, params.buf0_b, params.buf0_c, params.buf0_rhs,
        params.buf1_a, params.buf1_b, params.buf1_c, params.buf1_rhs, M, rFinal,
        bFinal);

    // Clamped BC: the solver produces all N values z_0..z_{N-1} directly
    // in rFinal[0..N-1], so zi and zi1 index straight in.
    for (int i = threadIdx.x; i < N - 1; i += blockDim.x) {
      T zi = rFinal[i] / bFinal[i];
      T zi1 = rFinal[i + 1] / bFinal[i + 1];
      set_spline_coeffs_from_tridiag_solution(splines[off + i], x[off + i],
                                              xWidth(&x[off], N, i), y[off + i],
                                              y[off + i + 1], zi, zi1);
    }
    __syncthreads();
  }
}

/**
 * @brief Kernel: fit periodic cubic splines for multiple rows.
 * One CUDA block per row. Each row has N data points (no duplicated endpoint),
 * producing N CubicSpline<T> intervals (the last wraps back to x[0]+xPeriod).
 *
 * Uses the cyclic tridiagonal solver (Sherman-Morrison), which requires
 * separate input arrays that survive across two PCR solves. Therefore the
 * caller must provide:
 *   - arr_a/b/c/rhs: storage for the arranged cyclic tridiagonal coefficients
 *   - buf0/buf1: PCR double-buffer workspace
 *   - z_out: output array for the solved z values
 *
 * @param x       Input x-coordinates, strided (row blk starts at blk*stride)
 * @param y       Input y-values, same layout
 * @param splines Output CubicSpline array, N per row (same stride)
 * @param arr_a    Arranged coeff storage, sub-diagonal (separate from
 * workspace)
 * @param arr_b    Arranged coeff storage, diagonal
 * @param arr_c    Arranged coeff storage, super-diagonal
 * @param arr_rhs  Arranged coeff storage, RHS
 * @param buf0_a    PCR workspace buffer 0, sub-diagonal
 * @param buf0_b    PCR workspace buffer 0, diagonal
 * @param buf0_c    PCR workspace buffer 0, super-diagonal
 * @param buf0_rhs  PCR workspace buffer 0, RHS
 * @param buf1_a    PCR workspace buffer 1, sub-diagonal
 * @param buf1_b    PCR workspace buffer 1, diagonal
 * @param buf1_c    PCR workspace buffer 1, super-diagonal
 * @param buf1_rhs  PCR workspace buffer 1, RHS
 * @param z_out   Output array for solved z values (length >= stride*num_rows)
 * @param num_points_in_row  Number of data points N per row
 * @param stride_elements_per_row  Stride between rows in all arrays
 * @param num_rows  Total number of rows
 * @param xPeriod  The period length
 */
template <typename T>
__global__ void periodic_spline_blockwise_kernel(
    const T *x, const T *y, CubicSpline<T> *splines, T *arr_a, T *arr_b,
    T *arr_c, T *arr_rhs, T *buf0_a, T *buf0_b, T *buf0_c, T *buf0_rhs,
    T *buf1_a, T *buf1_b, T *buf1_c, T *buf1_rhs, T *z_out,
    const size_t *num_points_in_row, const int stride_elements_per_row,
    const int num_rows, T xPeriod) {

  for (int blk = blockIdx.x; blk < num_rows; blk += gridDim.x) {
    int off = blk * stride_elements_per_row;
    int N = num_points_in_row[blk];

    // Arrange the cyclic tridiag system into arr_* arrays
    arrange_periodic_spline_blockwide(&x[off], &y[off], N, xPeriod, &arr_a[off],
                                      &arr_b[off], &arr_c[off], &arr_rhs[off]);
    __syncthreads();

    // Solve the cyclic system using Sherman-Morrison
    // arr_* are the persistent input arrays; buf0/buf1 are PCR workspace
    cutridiag::tridiag_pcr_scratch<T> ws_params = {
        &buf0_a[off],   &buf0_b[off],   &buf0_c[off],
        &buf0_rhs[off], &buf1_a[off],   &buf1_b[off],
        &buf1_c[off],   &buf1_rhs[off], (size_t)N};
    cutridiag::cyclic_tridiag_blockwide_pcr<T>(ws_params, &arr_a[off],
                                               &arr_b[off], &arr_c[off],
                                               &arr_rhs[off], &z_out[off]);

    // Periodic BC: the solver produces all N values z_0..z_{N-1} in z_out.
    // There are N intervals: N-1 normal ones plus the wrap-around interval.
    T hLast = xPeriod - (x[off + N - 1] - x[off]);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      T zi = z_out[off + i];
      T zi1 = z_out[off + (i + 1) % N];
      T h = (i < N - 1) ? xWidth(&x[off], N, i) : hLast;
      T yi = y[off + i];
      T yi1 = (i < N - 1) ? y[off + i + 1] : y[off];
      set_spline_coeffs_from_tridiag_solution(splines[off + i], x[off + i], h,
                                              yi, yi1, zi, zi1);
    }
    __syncthreads();
  }
}

} // namespace cuspline

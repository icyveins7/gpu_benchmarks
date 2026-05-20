#include <gtest/gtest.h>

#include "cpu_tridiag.h"
#include "tridiag.cuh"

#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cutridiag;

// Generate a random diagonally dominant tridiagonal system.
// num_rows rows, each with stride elements allocated but only
// row_lengths[row] active elements. row_lengths.size() == num_rows.
template <typename T>
void generate_tridiag_system(int num_rows, int stride,
                             const std::vector<int> &row_lengths,
                             thrust::host_vector<T> &ha,
                             thrust::host_vector<T> &hb,
                             thrust::host_vector<T> &hc,
                             thrust::host_vector<T> &hr) {
  ha.resize(num_rows * stride, T(0));
  hb.resize(num_rows * stride, T(0));
  hc.resize(num_rows * stride, T(0));
  hr.resize(num_rows * stride, T(0));

  for (int row = 0; row < num_rows; ++row) {
    int off = row * stride;
    int n = row_lengths[row];
    ha[off] = T(0);
    hb[off] = T(4) + T(rand()) / T(RAND_MAX);
    hc[off] = T(rand()) / T(RAND_MAX);
    hr[off] = T(rand()) / T(RAND_MAX);
    for (int i = 1; i < n - 1; ++i) {
      ha[off + i] = T(rand()) / T(RAND_MAX);
      hb[off + i] = T(4) + T(rand()) / T(RAND_MAX);
      hc[off + i] = T(rand()) / T(RAND_MAX);
      hr[off + i] = T(rand()) / T(RAND_MAX);
    }
    ha[off + n - 1] = T(rand()) / T(RAND_MAX);
    hb[off + n - 1] = T(4) + T(rand()) / T(RAND_MAX);
    hc[off + n - 1] = T(0);
    hr[off + n - 1] = T(rand()) / T(RAND_MAX);
  }
}

// Generic test: runs CPU tridag as reference, then tests both global-mem
// and shmem PCR kernels. Supports multiple rows with varying lengths.
template <typename T>
void test_pcr(int num_rows, int stride, const std::vector<int> &row_lengths,
              int num_threads, T tol) {
  srand(42);

  thrust::host_vector<T> ha, hb, hc, hr;
  generate_tridiag_system<T>(num_rows, stride, row_lengths, ha, hb, hc, hr);

  // CPU reference
  int max_n = *std::max_element(row_lengths.begin(), row_lengths.end());
  thrust::host_vector<T> hu_cpu(num_rows * stride, T(0));
  thrust::host_vector<T> hgam(max_n);
  for (int row = 0; row < num_rows; ++row) {
    int off = row * stride;
    int n = row_lengths[row];
    tridag<T>(&ha[off], &hb[off], &hc[off], &hr[off], &hu_cpu[off], hgam.data(),
              n);
  }

  // Prepare device data
  thrust::device_vector<T> da = ha, db = hb, dc = hc, dr = hr;
  thrust::device_vector<T> du(num_rows * stride, T(0));
  thrust::host_vector<size_t> hlen(num_rows);
  for (int i = 0; i < num_rows; ++i)
    hlen[i] = (size_t)row_lengths[i];
  thrust::device_vector<size_t> dlen = hlen;

  // --- Global memory PCR ---
  {
    TridiagPCRWorkspace<T> ws(num_rows, stride);
    tridiag_blockwise_pcr_kernel<T><<<num_rows, num_threads>>>(
        da.data().get(), db.data().get(), dc.data().get(), dr.data().get(),
        du.data().get(), ws.buf0_a_ptr(), ws.buf0_b_ptr(), ws.buf0_c_ptr(),
        ws.buf0_rhs_ptr(), ws.buf1_a_ptr(), ws.buf1_b_ptr(), ws.buf1_c_ptr(),
        ws.buf1_rhs_ptr(), dlen.data().get(), stride, num_rows);
    cudaDeviceSynchronize();

    thrust::host_vector<T> hu_gpu = du;
    for (int row = 0; row < num_rows; ++row) {
      int off = row * stride;
      int n = row_lengths[row];
      for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(hu_cpu[off + i], hu_gpu[off + i], tol)
            << "Global mem mismatch at row " << row << " index " << i;
      }
    }
  }

  // --- Shared memory PCR ---
  {
    thrust::fill(du.begin(), du.end(), T(0));
    size_t shmemBytes = TridiagPCRWorkspace<T>::requiredShmemBytes(stride);

    tridiag_blockwise_pcr_shmem_kernel<T>
        <<<num_rows, num_threads, shmemBytes>>>(
            da.data().get(), db.data().get(), dc.data().get(), dr.data().get(),
            du.data().get(), dlen.data().get(), stride, num_rows);
    cudaDeviceSynchronize();

    thrust::host_vector<T> hu_gpu = du;
    for (int row = 0; row < num_rows; ++row) {
      int off = row * stride;
      int n = row_lengths[row];
      for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(hu_cpu[off + i], hu_gpu[off + i], tol)
            << "Shmem mismatch at row " << row << " index " << i;
      }
    }
  }
}

// ---- Double tests ----

TEST(CudaTridiagDouble, SingleRowSimple) {
  test_pcr<double>(1, 4, {4}, 32, 1e-10);
}

TEST(CudaTridiagDouble, SingleRowLong) {
  test_pcr<double>(1, 64, {64}, 32, 1e-10);
}

TEST(CudaTridiagDouble, MultiRow) { test_pcr<double>(2, 4, {4, 4}, 32, 1e-10); }

TEST(CudaTridiagDouble, MultiRowLong) {
  test_pcr<double>(2, 64, {64, 64}, 32, 1e-10);
}

TEST(CudaTridiagDouble, MultiRowVarying) {
  test_pcr<double>(2, 64, {40, 50}, 32, 1e-10);
}

// ---- Float tests ----

TEST(CudaTridiagFloat, SingleRowSimple) {
  test_pcr<float>(1, 4, {4}, 32, 1e-5f);
}

TEST(CudaTridiagFloat, SingleRowLong) {
  test_pcr<float>(1, 64, {64}, 32, 1e-5f);
}

TEST(CudaTridiagFloat, MultiRow) { test_pcr<float>(2, 4, {4, 4}, 32, 1e-5f); }

TEST(CudaTridiagFloat, MultiRowLong) {
  test_pcr<float>(2, 64, {64, 64}, 32, 1e-5f);
}

TEST(CudaTridiagFloat, MultiRowVarying) {
  test_pcr<float>(2, 64, {40, 50}, 32, 1e-5f);
}

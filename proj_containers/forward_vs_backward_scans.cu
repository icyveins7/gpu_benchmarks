/*
This file proves that the backward scan is identical in speed to the forward
scan. This pattern should hold for any scan; InclusiveSumByKey is used here only
as an example.

The main point is to show that contiguous memory accesses still occur in the
backward scan, and hence there should be no performance impact (in fact during
the timings the backward scan was slightly faster, but this was negligible and
was likely noise).
*/

#include <cstdlib>
#include <iostream>

#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <containers/cubwrappers.cuh>

using IndexToRowFunctor = cubw::helpers::IndexToRowFunctor<int>;
using IndexToReverseRowFunctor = cubw::helpers::IndexToReverseRowFunctor<int>;

using FwdKeyIter = thrust::transform_iterator<IndexToRowFunctor,
                                              thrust::counting_iterator<int>>;
using RevKeyIter = thrust::transform_iterator<IndexToReverseRowFunctor,
                                              thrust::counting_iterator<int>>;

int main(int argc, char *argv[]) {
  int rows = 2;
  int cols = 8;
  if (argc > 1)
    rows = atoi(argv[1]);
  if (argc > 2)
    cols = atoi(argv[2]);
  int totalSize = rows * cols;

  printf("Forward vs backward scans: %d rows x %d cols = %d elements\n", rows,
         cols, totalSize);

  using DataT = int;

  // Generate random data on host
  thrust::host_vector<DataT> h_values(totalSize);
  for (int i = 0; i < totalSize; i++)
    h_values[i] = static_cast<DataT>(std::rand() % 10);

  thrust::device_vector<DataT> d_values = h_values;
  thrust::device_vector<DataT> d_out_fwd(totalSize);
  thrust::device_vector<DataT> d_out_bwd(totalSize);

  // --- Forward scan (left-to-right within each row) ---
  auto fwdKeyIter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{cols});

  cubw::DeviceScan::InclusiveSumByKey<FwdKeyIter, const DataT *, DataT *>
      fwdScan(totalSize);

  printf("Total forward scan cub workspace size: %zu\n",
         fwdScan.d_temp_storage.size());

  nvtxRangePushA("forward_scan");
  // repeat for warmup/nsys profile timing accuracy
  for (int iter = 0; iter < 2; ++iter) {
    fwdScan.exec(fwdKeyIter, d_values.data().get(), d_out_fwd.data().get(),
                 totalSize);
  }
  nvtxRangePop();

  // --- Backward scan (right-to-left within each row) ---
  auto revKeyIter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      IndexToReverseRowFunctor{cols, totalSize});

  const DataT *d_vals_ptr = d_values.data().get();
  DataT *d_out_bwd_ptr = d_out_bwd.data().get();
  thrust::reverse_iterator<const DataT *> rev_in(d_vals_ptr + totalSize);
  thrust::reverse_iterator<DataT *> rev_out(d_out_bwd_ptr + totalSize);

  cubw::DeviceScan::InclusiveSumByKey<RevKeyIter,
                                      thrust::reverse_iterator<const DataT *>,
                                      thrust::reverse_iterator<DataT *>>
      bwdScan(totalSize);
  printf("Total backward scan cub workspace size: %zu\n",
         bwdScan.d_temp_storage.size());

  nvtxRangePushA("backward_scan");
  // repeat for warmup/nsys profile timing accuracy
  for (int iter = 0; iter < 2; ++iter) {
    bwdScan.exec(revKeyIter, rev_in, rev_out, totalSize);
  }
  nvtxRangePop();

  // Print for verification at small sizes
  if (rows <= 2 && cols <= 8) {
    thrust::host_vector<DataT> h_fwd = d_out_fwd;
    thrust::host_vector<DataT> h_bwd = d_out_bwd;

    for (int r = 0; r < rows; r++) {
      printf("Row %d input:    ", r);
      for (int j = 0; j < cols; j++)
        printf("%2d ", h_values[r * cols + j]);
      printf("\n");

      printf("Row %d fwd scan: ", r);
      for (int j = 0; j < cols; j++)
        printf("%2d ", h_fwd[r * cols + j]);
      printf("\n");

      printf("Row %d bwd scan: ", r);
      for (int j = 0; j < cols; j++)
        printf("%2d ", h_bwd[r * cols + j]);
      printf("\n");
    }
  }

  return 0;
}

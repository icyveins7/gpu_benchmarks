#include <iostream>
#include <random>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "containers/cubwrappers.cuh"
#include "containers/image.cuh"
#include "transpose.cuh"

struct IndexToRowFunctor {
  int width;
  __host__ __device__ int operator()(int i) { return i / width; }
};

template <typename Tdata, typename Tidx, int BLOCK_THREADS>
__device__ void computeRowPrefixSums_BlockPerRow(
    containers::Image<Tdata, Tidx> image,
    containers::Image<Tdata, Tidx> rowSums,
    typename cub::BlockScan<Tdata, BLOCK_THREADS>::TempStorage &temp_storage) {
  using BlockScan = cub::BlockScan<Tdata, BLOCK_THREADS>;

  // Each block does one row
  for (int i = blockIdx.x; i < image.height; i += gridDim.x) {
    // Define a carryover for this row
    Tdata carryover = 0;

    // Read elements in the row
    for (int j = 0; j < image.width; j += blockDim.x) {
      Tdata tElem = 0;           // default value
      Tidx jt = j + threadIdx.x; // column for this thread
      if (image.colIsValid(jt)) {
        tElem = image.at(i, jt);
      }
      // CUB is great, no need to syncthreads since it performs internal warp
      // syncs
      Tdata newCarryover;
      BlockScan(temp_storage)
          .InclusiveSum(tElem, tElem, newCarryover); // in-place write

      // Add carryover
      tElem += carryover;

      // Also add the new carryover
      carryover += newCarryover;

      // Write back
      if (image.colIsValid(jt)) {
        rowSums.at(i, jt) = tElem;
      }

      // In order to reuse everything, we need to sync
      __syncthreads();
    }
  }
}

template <typename Tdata, typename Tidx, int BLOCK_THREADS>
__global__ void computeSATkernel(containers::Image<Tdata, Tidx> image,
                                 containers::Image<Tdata, Tidx> rowSums,
                                 containers::Image<Tdata, Tidx> sat) {
  using BlockScan = cub::BlockScan<Tdata, BLOCK_THREADS>;
  using TempStorage = typename BlockScan::TempStorage;

  __shared__ TempStorage temp_storage;

  computeRowPrefixSums_BlockPerRow<Tdata, Tidx, BLOCK_THREADS>(image, rowSums,
                                                               temp_storage);

  // TODO: do the columns
}

template <typename Tdata, typename Tidx>
__global__ void convolve_via_SAT_and_rowSums_naive_kernel(
    containers::Image<Tdata, Tidx> rowSums,
    containers::Image<Tdata, Tidx> sat) {

  return;
}

int main(int argc, char **argv) {
  printf("Summed area tables\n");

  int height = 16, width = 16;
  if (argc > 1) {
    height = atoi(argv[1]);
    width = atoi(argv[2]);
  }
  printf(" Image size (rows x columns): %d x %d\n", height, width);

  thrust::host_vector<int> h_data(height * width);
  for (int i = 0; i < height * width; i++) {
    h_data[i] = std::rand() % 10;
  }
  thrust::host_vector<int> h_rowSums(height * width);
  thrust::host_vector<int> h_sat(height * width);

  thrust::device_vector<int> d_data(h_data);
  thrust::device_vector<int> d_rowSums(h_data.size());
  thrust::device_vector<int> d_transpose(h_data.size());
  thrust::device_vector<int> d_sat(h_data.size());

  containers::Image<int, int> image(d_data.data().get(), width, height);
  containers::Image<int, int> rowSums(d_rowSums.data().get(), width, height);
  containers::Image<int, int> transposeImage(d_transpose.data().get(), height,
                                             width);
  containers::Image<int, int> sat(d_sat.data().get(), width, height);

  // CUB related prep
  auto rowKeyIterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{width});
  cubw::DeviceScan::InclusiveSumByKey<decltype(rowKeyIterator), int *, int *>
      cubwRowScan(width * height);

  auto colTransposeKeyIterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), IndexToRowFunctor{height});
  cubw::DeviceScan::InclusiveSumByKey<decltype(colTransposeKeyIterator), int *,
                                      int *>
      cubwColScanTranspose(height * width);

  // === 1. Perform prefix sums across rows

  // Custom kernel method (not enough occupancy to do this, must optimize beyond
  // 1 block per row)
  // {
  //   constexpr int tpb = 256;
  //   int blks = (height + tpb - 1) / tpb;
  //   computeSATkernel<int, int, tpb><<<blks, tpb>>>(image, rowSums, sat);
  // }

  // Generic CUB routine
  cubwRowScan.exec(rowKeyIterator, d_data.data().get(), d_rowSums.data().get(),
                   width * height);

  // === 2a. Perform prefix sums across columns (via explicitly transposed
  // matrix) Transpose into the SAT matrix
  transpose<int>(d_transpose.data().get(), d_rowSums.data().get(), (int)height,
                 (int)width);
  // In-place row-sums on the transpose (width is now the original height)
  cubwColScanTranspose.exec(colTransposeKeyIterator, d_transpose.data().get(),
                            d_transpose.data().get(), width * height);
  // Transposing back
  transpose<int>(d_sat.data().get(), d_transpose.data().get(), width, height);

  h_rowSums = d_rowSums;
  h_sat = d_sat;

  // Check row sums
  for (int i = 0; i < height; ++i) {
    int sum = 0;
    for (int j = 0; j < width; ++j) {
      sum += h_data[i * width + j];
      if (sum != h_rowSums[i * width + j]) {
        printf("Mismatch at (%d, %d): expected %d vs %d\n", i, j, sum,
               h_rowSums[i * width + j]);
        break;
      }
    }
  }

  if (height <= 32 && width <= 32) {
    printf("-------- Original:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4d ", h_data[i * width + j]);
      }
      printf("\n");
    }

    printf("-------- Row sums:\n");
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        printf("%4d ", h_rowSums[i * width + j]);
      }
      printf("\n");
    }
  }

  return 0;
}

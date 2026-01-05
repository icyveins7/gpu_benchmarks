#include <iostream>
#include <random>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "containers/image.cuh"

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
  thrust::device_vector<int> d_sat(h_data.size());

  containers::Image<int, int> image(d_data.data().get(), width, height);
  containers::Image<int, int> rowSums(d_rowSums.data().get(), width, height);
  containers::Image<int, int> sat(d_sat.data().get(), width, height);

  constexpr int tpb = 256;
  int blks = (height + tpb - 1) / tpb;
  computeSATkernel<int, int, tpb><<<blks, tpb>>>(image, rowSums, sat);

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

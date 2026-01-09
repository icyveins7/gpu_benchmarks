#pragma once

/**
 * @brief Copied/adapted from Harris's example on
 * https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 *
 * @tparam T Type of data
 * @param odata Output matrix, should not overlap the input
 * @param idata Input matrix
 * @param rows Number of rows of either matrix
 * @param cols Number of cols of either matrix
 */
template <typename T, int TILE_DIM = 32>
__global__ void transposeKernel(T *odata, const T *idata, const int rows,
                                const int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += blockDim.y) {
    // Ensure we don't read out of range
    if (x < cols && y + j < rows)
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * cols + x];
    else
      tile[threadIdx.y + j][threadIdx.x] = 0;
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += blockDim.y)
    // Write only in range
    if (x < cols && y + j < rows)
      odata[(y + j) * cols + x] = tile[threadIdx.x][threadIdx.y + j];
}

template <typename T, int TILE_DIM = 32>
void transpose(T *odata, const T *idata, const int rows, const int cols,
               cudaStream_t stream = 0) {
  constexpr dim3 tpb(32, 4);
  dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
  transposeKernel<<<grid, tpb, 0, stream>>>(odata, idata, rows, cols);
}

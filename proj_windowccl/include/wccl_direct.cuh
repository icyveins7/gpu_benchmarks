/*
 * For small problems, we simply do the O(N^2) checks.
 * That is, for a given pixel i, attempt to unite it with all pixels [0, i-1].
 * The unite possibilities can be visualized by the lower triangle of a square matrix,
 * not including the diagonal (since there's no point in uniting with itself).
 *
 * E.g. for 5x5,
 * - - - - -
 * O - - - -
 * O O - - -
 * O O O - -
 * O O O O -
 *
 * That is, for each target pixel (row), we check the possible pixels (columns with O) to unite.
 */

#pragma once

template <typename Txy = short2, typename Tidx = int>
__global__ void unite_pixel_locations_direct_kernel(
  const Txy* xy,
  const int numPixels,
  const Txy windowDist,
  const int2 numPerTile, // should aim to have numPerTile.y >= threads per block
  const int2 numTiles,
  Tidx* labels
){
  for (int blki = blockIdx.x; blki < numTiles.y; blki += gridDim.x){
    int startRow = blki * numPerTile.y;
    for (int blkj = blockIdx.y; blkj < numTiles.x; blkj += gridDim.y){
      int startCol = blkj * numPerTile.x;

      // Load the pixels being operated on into shared memory
      extern __shared__ Txy sharedMem[];
      Txy* s_xy_rows = sharedMem; // length numPerTile.y
      Txy* s_xy_cols = s_xy_rows + numPerTile.y; // length numPerTile.x
      Tidx s_idx = s_xy_cols + numPerTile.x; // length numPerTile.y i.e. rows

      int t = threadIdx.y * blockDim.x + threadIdx.x;
      for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < numPerTile.y; t += blockDim.x * gridDim.x){
        s_xy_rows[t] = xy[startRow + t];
      }
      for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < numPerTile.x; t += blockDim.x * gridDim.x){
        s_xy_cols[t] = xy[startCol + t];
      }
      for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < numPerTile.y; t += blockDim.x * gridDim.x){
        s_idx[t] = startRow + t;
      }
      __syncthreads();

      // Do comparisons
      for (int i = threadIdx.y; i < numPerTile.y; i += blockDim.y){
        for (int j = threadIdx.x; j < numPerTile.x; j += blockDim.x){
          Txy dist = make_int2(
            abs(s_xy_rows[i].x - s_xy_cols[j].x),
            abs(s_xy_rows[i].y - s_xy_cols[j].y)
          );
          if (dist.x <= windowDist.x && dist.y <= windowDist.y){
            // If the distance is less than the window size, then we can unite
            atomicMin(&s_idx[i], startCol + j);
          }
        }
      }

      // Write to final labels
      for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < numPerTile.y; t += blockDim.x * gridDim.x){
        atomicMin(&labels[startRow + t], s_idx[t]);
      }
    }
  }
}

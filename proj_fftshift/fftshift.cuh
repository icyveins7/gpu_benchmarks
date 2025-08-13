#include <thrust/complex.h>

template <typename T, typename U = T, bool norm = false>
__global__ void fftShift2D_kernel(const T *in, U *out, int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int widthInc = width % 2 == 0 ? width / 2 : (width + 1) / 2;
  int heightInc = height % 2 == 0 ? height / 2 : (height + 1) / 2;

  if (x < width && y < height) {
    // Compute the new positions after shifting
    int newX = (widthInc + x) % width;
    int newY = (heightInc + y) % height;

    if constexpr (norm)
      out[y * width + x] = thrust::norm(in[newY * width + newX]);
    else
      out[y * width + x] = in[newY * width + newX];
  }
}

template <typename T, typename U = T, bool norm = false>
void fftShift2D(const T *d_data, U *d_out, int width, int height,
                dim3 threadsPerBlock = {16, 16}) {
  // The kernel expects that the entire matrix is covered, no grid strides
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  printf("Blks: %d %d\n", blocksPerGrid.x, blocksPerGrid.y);

  fftShift2D_kernel<T, U, norm>
      <<<blocksPerGrid, threadsPerBlock>>>(d_data, d_out, width, height);
}

#include "transpose.cuh"
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char **argv) {
  int rows = 32, cols = 32;
  if (argc == 3) {
    rows = std::atoi(argv[1]);
    cols = std::atoi(argv[2]);
  }
  printf("input rows = %d, cols = %d\n", rows, cols);

  thrust::host_vector<int> h_a(rows * cols);
  thrust::device_vector<int> d_a(rows * cols);
  thrust::host_vector<int> h_b(rows * cols);
  thrust::device_vector<int> d_b(rows * cols);
  thrust::device_vector<int> d_copy(rows * cols);

  for (int i = 0; i < rows * cols; ++i) {
    h_a[i] = std::rand() % 10;
  }

  d_a = h_a;
  transpose<int>(d_b.data().get(), d_a.data().get(), rows, cols);
  h_b = d_b;

  // Copy for comparison
  d_copy = d_b;

  if (rows <= 64 && cols <= 64) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%d ", h_a[i * cols + j]);
      }
      printf("\n");
    }
    printf("\n =========== \n");
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        printf("%d ", h_b[i * rows + j]);
      }
      printf("\n");
    }
    printf("\n =========== \n");
  }

  // Checks
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (h_a[i * cols + j] != h_b[j * rows + i]) {
        printf("ERROR: h_a[%d][%d] = %d, h_b[%d][%d] = %d\n", i, j,
               h_a[i * cols + j], j, i, h_b[j * rows + i]);
        return 1;
      }
    }
  }

  return 0;
}

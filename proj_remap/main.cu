#include "kernels.h"
#include <iostream>
#include <random>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  long int THREADS_PER_BLK;
  char *p;
  if (argc >= 2)
    THREADS_PER_BLK = strtol(argv[1], &p, 10);
  else
    THREADS_PER_BLK = 16;
  printf("Using THREADS_PER_BLK (per dim) = %ld\n", THREADS_PER_BLK);

  size_t width, height;
  if (argc >= 4) {
    width = strtol(argv[2], &p, 10);
    height = strtol(argv[3], &p, 10);
  } else {
    width = 128;
    height = 128;
  }
  printf("Using width = %ld, height = %ld\n", width, height);

  // Generate some requested input
  thrust::host_vector<float> h_x(width * height);
  thrust::host_vector<float> h_y(width * height);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (float &in : h_x)
    in = dis(gen) * width;
  for (float &in : h_y)
    in = dis(gen) * height;

  thrust::device_vector<float> d_x(width * height);
  thrust::device_vector<float> d_y(width * height);
  d_x = h_x; // copy input to device
  d_y = h_y;
  // Allocate output as well
  thrust::device_vector<float> d_out(width * height);
  thrust::host_vector<float> h_out(width * height);

  // Test NaiveRemap
  NaiveRemap<float> nr(height, width);
  nr.set_tpb(THREADS_PER_BLK);
  printf("Device src size = %zd\n", nr.get_d_src().size());
  printf("Host src size = %zd\n", nr.get_h_src().size());
  // quickView(nr.get_h_src(), width, height);
  printf("Expected size = %zd\n", height * width);
  for (int i = 0; i < 3; ++i) {
    nr.d_run(d_x, d_y, width, height, d_out);
    printf("Ran NaiveRemap (%d)\n", i);
  }
  h_out = d_out;
  // quickView(h_out, width, height);

  return 0;
}

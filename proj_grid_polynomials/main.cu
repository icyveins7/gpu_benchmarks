#include "kernels.h"
#include <iostream>

int main() {
  // Create some coefficients
  NaiveGridPolynom<float> gp(10);
  // See some of the parameters
  printf("Coefficients: ");
  for (auto coeff : gp.h_coeffs()) {
    printf("%f\n", coeff);
  }

  // Run the kernel?
  const size_t length = 40000000;
  thrust::device_vector<float> d_in(length);
  thrust::device_vector<float> d_out(length);

  const int repeats = 3;

  for (int i = 0; i < repeats; ++i)
    gp.d_run(d_in, d_out);

  return 0;
}

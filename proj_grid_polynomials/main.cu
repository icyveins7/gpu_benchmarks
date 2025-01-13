#include "kernels.h"
#include <iostream>

int main() {
  // Initialize some input/output
  const size_t length = 40000000;
  // Randomize some input on host
  thrust::host_vector<float> h_in(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (float &in : h_in)
    in = dis(gen);

  thrust::device_vector<float> d_in(length);
  d_in = h_in; // copy input to device
  thrust::device_vector<float> d_out(length);

  // Repeat each kernel a few times for good measure
  const int repeats = 3;
  const size_t numCoeffs = 10;
  // ============ Naive kernel test ================
  NaiveGridPolynom<float> gp(numCoeffs);
  // See some of the parameters
  printf("Coefficients: ");
  for (auto coeff : gp.h_coeffs()) {
    printf("%f\n", coeff);
  }

  for (int i = 0; i < repeats; ++i)
    gp.d_run(d_in, d_out);
  // ============ Shared mem coeffs kernel test ================
  SharedCoeffGridPolynom<float> scp(numCoeffs);

  for (int i = 0; i < repeats; ++i)
    scp.d_run(d_in, d_out);

  return 0;
}

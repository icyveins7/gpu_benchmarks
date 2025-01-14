#include "kernels.h"
#include <iostream>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  long int THREADS_PER_BLK;
  char *p;
  if (argc > 1)
    THREADS_PER_BLK = strtol(argv[1], &p, 10);
  else
    THREADS_PER_BLK = 128;
  printf("Using THREADS_PER_BLK = %ld\n", THREADS_PER_BLK);

  // Initialize some input/output
  const size_t length = 15000000; // 40000000;
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
  const size_t numCoeffs = 12;
  // ============ Naive kernel test ================
  NaiveGridPolynom<float> gp(numCoeffs);
  gp.set_tpb(THREADS_PER_BLK);
  // See some of the parameters
  printf("Coefficients: \n");
  for (auto coeff : gp.h_coeffs()) {
    printf("%f\n", coeff);
  }

  for (int i = 0; i < repeats; ++i)
    gp.d_run(d_in, d_out);
  // ============ Shared mem coeffs kernel test ================
  SharedCoeffGridPolynom<float> scp(numCoeffs);
  scp.set_tpb(THREADS_PER_BLK);

  for (int i = 0; i < repeats; ++i)
    scp.d_run(d_in, d_out);
  // ============ Constant mem coeffs kernel test ================
  ConstantCoeffGridPolynom<float> ccp(numCoeffs);
  ccp.set_tpb(THREADS_PER_BLK);

  for (int i = 0; i < repeats; ++i)
    ccp.d_run(d_in, d_out);

  // ===================================================
  // ============ Sin Series Tests =====================
  // ===================================================
  const size_t sinNumCoeffs = 5;

  // ============ Naive kernel test ================
  NaiveGridSinSeries<float> ss(sinNumCoeffs);
  ss.set_tpb(THREADS_PER_BLK);

  for (int i = 0; i < repeats; ++i)
    ss.d_run(d_in, d_out);

  // ============ Intrinsics kernel test ================
  IntrinsicGridSinSeries<float> is(sinNumCoeffs);
  is.set_tpb(THREADS_PER_BLK);

  for (int i = 0; i < repeats; ++i)
    is.d_run(d_in, d_out);

  return 0;
}

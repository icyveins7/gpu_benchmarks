#include "patternmatching.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char *argv[]) {
  printf("Pattern matching tests: (length) (numInputs) (numPatterns)\n");

  // Some parameters with defaults
  const unsigned int length = argc >= 2 ? strtol(argv[1], nullptr, 10) : 64;
  const unsigned int numInputs =
      argc >= 3 ? strtol(argv[2], nullptr, 10) : 1024;
  const unsigned int numPatterns =
      argc >= 4 ? strtol(argv[3], nullptr, 10) : 4096;
  const unsigned int repeats = 3;

  // Make some random vectors
  thrust::host_vector<float> h_inputs(numInputs * length);
  thrust::host_vector<float> h_patterns(numPatterns * length);

  thrust::generate(h_inputs.begin(), h_inputs.end(), rand);
  thrust::generate(h_patterns.begin(), h_patterns.end(), rand);

  thrust::device_vector<float> d_inputs = h_inputs;
  thrust::device_vector<float> d_patterns = h_patterns;
  thrust::device_vector<float> d_metric(numInputs * numPatterns);

  // Call the kernel
  const int threadsPerBlk = 64;
  const dim3 numBlks = {numPatterns, numInputs, 1};
  const int shmReq = length * sizeof(float);

  for (unsigned int i = 0; i < repeats; i++)
    naivePatternMatchKernel_LS<float, float>
        <<<numBlks, threadsPerBlk, shmReq>>>(
            thrust::raw_pointer_cast(d_inputs.data()), numInputs,
            thrust::raw_pointer_cast(d_patterns.data()), numPatterns, length,
            thrust::raw_pointer_cast(d_metric.data()));

  // Copy back results
  thrust::host_vector<float> h_metric = d_metric;

  // // Print results
  // for (int i = 0; i < numInputs; i++) {
  //   for (int j = 0; j < numPatterns; j++) {
  //     printf("%.2f ", h_metric[i * numPatterns + j]);
  //   }
  //   printf("\n");
  // }

  return 0;
}

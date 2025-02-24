#include "hist.cuh"

#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {

  float values[32] = {
      0.97871922, 0.25221157, 0.77178848, 0.78858679, 0.1512728,  0.8590046,
      0.50219307, 0.67468223, 0.50129885, 0.42735645, 0.02436939, 0.96941367,
      0.17397098, 0.11099226, 0.74936965, 0.13412934, 0.06048629, 0.89781347,
      0.93335581, 0.57877258, 0.39763006, 0.48534404, 0.74600632, 0.80596513,
      0.651098,   0.17545939, 0.04825435, 0.3180992,  0.73348803, 0.3253478,
      0.2855713,  0.01210941};

  thrust::device_vector<float> d_input(values, values + 32);
  thrust::device_vector<int> d_hist(32, 0);

  histogramKernel<float><<<1, 32>>>(d_input.data().get(), (int)d_input.size(),
                                    d_hist.data().get(), (int)d_hist.size(),
                                    0.03f, 0.0f);

  int expected[32] = {2, 1, 1, 1, 1, 3, 0, 0, 1, 1, 2, 0, 0, 1, 1, 0,
                      3, 0, 0, 1, 0, 1, 1, 0, 3, 1, 2, 0, 1, 1, 0, 1};
  thrust::host_vector<int> h_hist = d_hist;

  for (int i = 0; i < 32; i++) {
    if (h_hist[i] != expected[i]) {
      printf("%d: Expected: %d, got: %d\n", i, expected[i], h_hist[i]);
    }
  }

  // Test rough timings
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  thrust::host_vector<float> h_input(1000000);

  for (int i = 0; i < 1000000; i++) {
    h_input[i] = distribution(generator);
  }

  thrust::device_vector<float> d_input2(h_input);
  thrust::device_vector<int> d_hist2(8192, 0);

  histogramKernel<float><<<8192 / 256, 256>>>(
      d_input2.data().get(), (int)d_input2.size(), d_hist2.data().get(),
      (int)d_hist2.size(), 1.0f / 8192.0f, 0.0f);

  return 0;
}

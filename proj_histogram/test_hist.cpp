#include <gtest/gtest.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "hist.cuh"
#include "general.h"

template <typename T>
using numpyarray = py::array_t<T, py::array::c_style | py::array::forcecast>;

TEST(proj_histogram, testAgainstCpp) {
  // Generate some random data
  numpyarray<double> arr = pct::runPythonFunction("test_hist", "rand", 100);
  std::vector<double> vec = pct::fromNumpyArray<double>(arr);
  // Generate the bins
  numpyarray<double> bins = pct::runPythonFunction("numpy", "linspace", 0, 1, 101);

  py::tuple result = pct::runPythonFunction("numpy", "histogram", arr, bins);

  auto counts = result[0].cast<numpyarray<int>>();

  std::vector<int> vec_counts = pct::fromNumpyArray<int>(counts);

  // manually bin them
  std::vector<int> vec_countcheck(vec_counts.size());
  for (int i = 0; i < 100; i++) {
    int bin = int(vec[i] / 0.01);
    if (bin >= 0 && bin < 100)
      vec_countcheck[bin]++;
  }

  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(vec_countcheck[i], vec_counts[i]);
  }
}

TEST(proj_histogram, testAgainstCuda) {
  // Generate some random data
  numpyarray<double> arr = pct::runPythonFunction("test_hist", "rand", 1000);
  std::vector<double> vec = pct::fromNumpyArray<double>(arr);
  // Generate the bins
  numpyarray<double> bins = pct::runPythonFunction("numpy", "linspace", 0, 1, 101);
  std::vector<double> vec_bins = pct::fromNumpyArray<double>(bins);

  py::tuple result = pct::runPythonFunction("numpy", "histogram", arr, bins);

  auto counts = result[0].cast<numpyarray<int>>();

  // Run digitize to get bin indices of each input element
  auto indices = pct::runPythonFunction("numpy", "digitize", arr, bins);

  std::vector<int> vec_counts = pct::fromNumpyArray<int>(counts);
  std::vector<int> vec_indices = pct::fromNumpyArray<int>(indices);

  // manually bin them using the kernel
  thrust::device_vector<int> d_hist(counts.size());
  thrust::device_vector<int> d_binIndices(vec.size());
  thrust::device_vector<double> d_vec(vec.begin(), vec.end());
  histogramKernel<double, true><<<vec.size() / 32 + 1, 32>>>(
    d_vec.data().get(), (int)d_vec.size(), d_hist.data().get(), (int)d_hist.size(),
    1.0/100.0, 0.0, d_binIndices.data().get());

  thrust::host_vector<int> vec_countcheck = d_hist;
  thrust::host_vector<int> h_binIndices = d_binIndices;

  for (size_t i = 0; i < vec_countcheck.size(); i++) {
    printf("%zd: %f\n", i, vec[i]);
    int cudabin = h_binIndices[i];
    int numpybin = vec_indices[i];
    printf("Numpy bin: [%f, %f)\n", vec_bins[numpybin-1], vec_bins[numpybin]); // the digitize returns +1 bins
    printf("Cuda bin: [%f, %f)\n", cudabin * 0.01, (cudabin+1) * 0.01);
    EXPECT_EQ(vec_countcheck[i], vec_counts[i]);
    EXPECT_EQ(cudabin, numpybin-1);
  }
}

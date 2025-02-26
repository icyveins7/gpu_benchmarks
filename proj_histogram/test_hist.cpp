#include <gtest/gtest.h>
#include <vector>

#include "general.h"

template <typename T>
using numpyarray = py::array_t<T, py::array::c_style | py::array::forcecast>;

TEST(proj_histogram, test) {
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

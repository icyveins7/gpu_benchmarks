#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <thrust/fill.h>

#include "containers/image.cuh"
#include "overlapKernels.cuh"

#include "cxxopts.hpp"

/**
 * @brief Smoke-test four complete input pixels against one larger output.
 *
 * The output covers all four input pixels with area weight 0.25 each. Values
 * 10, 20, 30, 40 must average to 25 and clear the 0.45 validity threshold.
 * Return a nonzero exit code if the CUDA launch or expected result fails.
 */
int runSmoke() {
  containers::DeviceImageStorage<float> input(2, 2);
  thrust::host_vector<float> values(4);
  values[0] = 10;
  values[1] = 20;
  values[2] = 30;
  values[3] = 40;
  input.vec = values;
  containers::DeviceImageStorage<float> output(1, 1);

  overlap::remapOverlap(input.cimage(), output.image(),
                 overlap::OutputToInput<double>{{2, 0}, {0, 2}, {0.5, 0.5}},
                 0.45);
  const auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << cudaGetErrorString(status) << '\n';
    return 1;
  }
  thrust::host_vector<float> result = output.vec;
  std::cout << "2x2 to 1x1 area-weighted value: " << result[0] << '\n';
  return std::abs(result[0] - 25.0f) < 1e-5f ? 0 : 1;
}

/**
 * @brief Run the small smoke test, or profile a configurable affine remap.
 *
 * Matrix columns (axx, axy) and (ayx, ayy) and translation (tx, ty) map
 * output-pixel centres to input-pixel coordinates. Input values are all 1,
 * so every valid output must also equal 1. A warm-up precedes timed launches.
 */
int main(int argc, char* argv[]) {
  if (argc == 1)
    return runSmoke();

  cxxopts::Options options("overlap_remap",
                           "Output-driven CUDA pixel-overlap benchmark");
  options.add_options()
      ("inwidth", "Input image width", cxxopts::value<int>()->default_value("2"))
      ("inheight", "Input image height", cxxopts::value<int>()->default_value("2"))
      ("outwidth", "Output image width", cxxopts::value<int>()->default_value("1"))
      ("outheight", "Output image height", cxxopts::value<int>()->default_value("1"))
      ("axx", "Input X change per output X", cxxopts::value<double>()->default_value("2"))
      ("axy", "Input Y change per output X", cxxopts::value<double>()->default_value("0"))
      ("ayx", "Input X change per output Y", cxxopts::value<double>()->default_value("0"))
      ("ayy", "Input Y change per output Y", cxxopts::value<double>()->default_value("2"))
      ("tx", "Input X at output centre (0,0)", cxxopts::value<double>()->default_value("0.5"))
      ("ty", "Input Y at output centre (0,0)", cxxopts::value<double>()->default_value("0.5"))
      ("minweight", "Strict minimum output-area weight", cxxopts::value<double>()->default_value("0.45"))
      ("iterations", "Number of timed launches after one warm-up",
       cxxopts::value<int>()->default_value("3"))
      ("h,help", "Show options");
  const auto args = options.parse(argc, argv);
  if (args.count("help")) {
    std::cout << options.help() << '\n';
    return 0;
  }

  const int inWidth = args["inwidth"].as<int>();
  const int inHeight = args["inheight"].as<int>();
  const int outWidth = args["outwidth"].as<int>();
  const int outHeight = args["outheight"].as<int>();
  const int iterations = args["iterations"].as<int>();
  if (inWidth <= 0 || inHeight <= 0 || outWidth <= 0 || outHeight <= 0 ||
      iterations <= 0)
    throw std::invalid_argument("image dimensions and iterations must be positive");

  const overlap::OutputToInput<double> transform{
      {args["axx"].as<double>(), args["axy"].as<double>()},
      {args["ayx"].as<double>(), args["ayy"].as<double>()},
      {args["tx"].as<double>(), args["ty"].as<double>()}};
  const double minWeight = args["minweight"].as<double>();
  containers::DeviceImageStorage<float> input(inWidth, inHeight);
  containers::DeviceImageStorage<float> output(outWidth, outHeight);
  thrust::fill(input.vec.begin(), input.vec.end(), 1.0f);
  const auto launch = [&] {
    overlap::remapOverlap(input.cimage(), output.image(), transform, minWeight);
  };

  launch();
  if (const auto status = cudaDeviceSynchronize(); status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i)
    launch();
  cudaEventRecord(stop);
  if (const auto status = cudaEventSynchronize(stop); status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
  float elapsedMs = 0;
  cudaEventElapsedTime(&elapsedMs, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  thrust::host_vector<float> result = output.vec;
  const int valid = std::count_if(result.begin(), result.end(), [](float value) {
    return value != std::numeric_limits<float>::max();
  });
  const bool correct = std::all_of(result.begin(), result.end(), [](float value) {
    return value == std::numeric_limits<float>::max() ||
           std::abs(value - 1.0f) < 1e-4f;
  });
  std::cout << "Input " << inWidth << 'x' << inHeight << ", output " << outWidth
            << 'x' << outHeight << ", transform [[" << transform.xAxis.x << ", "
            << transform.yAxis.x << ", " << transform.origin.x << "], ["
            << transform.xAxis.y << ", " << transform.yAxis.y << ", "
            << transform.origin.y << "]], minweight " << minWeight << '\n'
            << "Valid outputs: " << valid << '/' << result.size() << '\n'
            << "CUDA event average: " << elapsedMs / iterations << " ms over "
            << iterations << " timed launches\n";
  return correct ? 0 : 1;
}

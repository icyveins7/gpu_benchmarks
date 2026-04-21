#include <iostream>

#include "containers/image.cuh"
#include "oversampleKernels.cuh"
#include "pinnedalloc.cuh"

#include "cxxopts.hpp"

int main(int argc, char *argv[]) {
  // Parse command line args
  // clang-format off
  cxxopts::Options options("Oversampled remap", "Cuda experiment of oversampled remap kernel");
  options.add_options()
    ("inheight", "Input height", cxxopts::value<int>()->default_value("4"))
    ("inwidth", "Input width", cxxopts::value<int>()->default_value("4"))
    ("outheight", "Output height", cxxopts::value<int>()->default_value("5"))
    ("outwidth", "Output width", cxxopts::value<int>()->default_value("5"))
    ("f,factor", "Oversample factor", cxxopts::value<int>()->default_value("3"))
    ("xoffset", "Output x offset", cxxopts::value<float>()->default_value("0"))
    ("yoffset", "Output y offset", cxxopts::value<float>()->default_value("0"))
    ("xstep", "Output x step", cxxopts::value<float>()->default_value("0.8"))
    ("ystep", "Output y step", cxxopts::value<float>()->default_value("0.8"))
    ("shm", "Use shared mem", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage")
  ;
  // clang-format on

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  bool useSharedMem = result["shm"].as<bool>();
  printf("Using shared mem? %s\n", useSharedMem ? "true" : "false");
  if (useSharedMem) {
    printf("Shared mem version not implemented yet\n");
    return -1;
  }

  containers::DeviceImageStorage<int> d_in(result["inheight"].as<int>(),
                                           result["inwidth"].as<int>());
  thrust::pinned_host_vector<int> h_in(d_in.vec.size());
  std::iota(h_in.begin(), h_in.end(), 1);
  d_in.vec = h_in;

  containers::DeviceImageStorage<float> d_out(result["outheight"].as<int>(),
                                              result["outwidth"].as<int>());
  int2 oversampleFactor{result["factor"].as<int>(), result["factor"].as<int>()};
  float2 outOffset{result["xoffset"].as<float>(),
                   result["yoffset"].as<float>()};
  float2 outStep{result["xstep"].as<float>(), result["ystep"].as<float>()};
  using Tcalc = float;
  if (useSharedMem) {
    oversampleBilerpAndCombine<int, float, Tcalc, true>(
        d_in.cimage(), d_out.image(), oversampleFactor, outOffset, outStep,
        dim3(32, 4));
  } else {
    oversampleBilerpAndCombine<int, float, Tcalc, false>(
        d_in.cimage(), d_out.image(), oversampleFactor, outOffset, outStep,
        dim3(32, 4));
  }
  thrust::pinned_host_vector<float> h_out = d_out.vec;

  if (d_in.width < 32 && d_in.height < 32) {
    for (size_t y = 0; y < d_in.height; y++) {
      for (size_t x = 0; x < d_in.width; x++) {
        size_t idx = y * d_in.width + x;
        printf("%2d ", h_in[idx]);
      }
      std::cout << std::endl;
    }
  }
  std::cout << "-------------------" << std::endl;

  if (d_out.width < 32 && d_out.height < 32) {
    for (size_t y = 0; y < d_out.height; y++) {
      for (size_t x = 0; x < d_out.width; x++) {
        size_t idx = y * d_out.width + x;
        printf("%8.3f ", h_out[idx]);
      }
      std::cout << std::endl;
    }
  }

  return 0;
}

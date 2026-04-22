#include <cmath>
#include <fstream>
#include <iostream>

#include "containers/image.cuh"
#include "oversampleKernels.cuh"
#include "pinnedalloc.cuh"

#include "cxxopts.hpp"

#if defined(USE_DOUBLE_CALC)
using Tcalc = double;
#else
using Tcalc = float;
#endif

int main(int argc, char *argv[]) {
#if defined(USE_DOUBLE_CALC)
  printf("Using Tcalc = double\n");
#endif
  // Parse command line args
  // clang-format off
  cxxopts::Options options("Oversampled remap", "Cuda experiment of oversampled remap kernel");
  options.add_options()
    ("inheight", "Input height", cxxopts::value<int>()->default_value("4"))
    ("inwidth", "Input width", cxxopts::value<int>()->default_value("4"))
    ("outheight", "Output height", cxxopts::value<int>()->default_value("5"))
    ("outwidth", "Output width", cxxopts::value<int>()->default_value("5"))
    ("f,factor", "Oversample factor", cxxopts::value<int>()->default_value("3"))
    ("xoffset", "Output x offset", cxxopts::value<Tcalc>()->default_value("0"))
    ("yoffset", "Output y offset", cxxopts::value<Tcalc>()->default_value("0"))
    ("xstep", "Output x step", cxxopts::value<Tcalc>()->default_value("0.8"))
    ("ystep", "Output y step", cxxopts::value<Tcalc>()->default_value("0.8"))
    ("shm", "Use shared mem", cxxopts::value<bool>()->default_value("false"))
    ("angle", "Rotation angle (degrees)", cxxopts::value<Tcalc>()->default_value("0"))
    ("o,output", "Output file", cxxopts::value<std::string>())
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

  containers::DeviceImageStorage<int> d_in(result["inheight"].as<int>(),
                                           result["inwidth"].as<int>());
  thrust::pinned_host_vector<int> h_in(d_in.vec.size());
  std::iota(h_in.begin(), h_in.end(), 1);
  d_in.vec = h_in;

  // We use Tcalc for the output as well
  containers::DeviceImageStorage<Tcalc> d_out(result["outheight"].as<int>(),
                                              result["outwidth"].as<int>());
  int2 oversampleFactor{result["factor"].as<int>(), result["factor"].as<int>()};
  cuda_vec2_t<Tcalc> outOffset{result["xoffset"].as<Tcalc>(),
                               result["yoffset"].as<Tcalc>()};
  cuda_vec2_t<Tcalc> outStep{result["xstep"].as<Tcalc>(),
                             result["ystep"].as<Tcalc>()};

  Tcalc angleRadians = result["angle"].as<Tcalc>() / 180.0 * M_PI;
  if (useSharedMem) {
    oversampleBilerpAndCombine<int, Tcalc, Tcalc, true>(
        d_in.cimage(), d_out.image(), oversampleFactor, outOffset, outStep,
        dim3(32, 4), angleRadians);
  } else {
    oversampleBilerpAndCombine<int, Tcalc, Tcalc, false>(
        d_in.cimage(), d_out.image(), oversampleFactor, outOffset, outStep,
        dim3(32, 4), angleRadians);
  }
  thrust::pinned_host_vector<Tcalc> h_out = d_out.vec;

  if (d_in.width < 64 && d_in.height < 64) {
    for (size_t y = 0; y < (size_t)d_in.height; y++) {
      for (size_t x = 0; x < (size_t)d_in.width; x++) {
        size_t idx = y * d_in.width + x;
        printf("%2d ", h_in[idx]);
      }
      std::cout << std::endl;
    }
  }
  std::cout << "-------------------" << std::endl;

  if (d_out.width < 64 && d_out.height < 64) {
    for (size_t y = 0; y < (size_t)d_out.height; y++) {
      for (size_t x = 0; x < (size_t)d_out.width; x++) {
        size_t idx = y * d_out.width + x;
        printf("%8.3f ", h_out[idx]);
      }
      std::cout << std::endl;
    }
  }
  if (result.count("output")) {
    std::ofstream out(result["output"].as<std::string>(), std::ios::binary);
    out.write((char *)h_out.data().get(), h_out.size() * sizeof(Tcalc));
  }

  return 0;
}

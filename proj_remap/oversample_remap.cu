#include <fstream>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>

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
    ("minweight", "Minimum valid samples", cxxopts::value<int>()->default_value("1"))
    ("kernel", "Interpolation kernel", cxxopts::value<std::string>()->default_value("bicubic"))
    ("xoffset", "Output x offset", cxxopts::value<Tcalc>()->default_value("0"))
    ("yoffset", "Output y offset", cxxopts::value<Tcalc>()->default_value("0"))
    ("xstep", "Output x step", cxxopts::value<Tcalc>()->default_value("0.8"))
    ("ystep", "Output y step", cxxopts::value<Tcalc>()->default_value("0.8"))
    ("shm", "Use shared mem", cxxopts::value<bool>()->default_value("false"))
    ("angle", "Rotation angle (degrees)", cxxopts::value<Tcalc>()->default_value("0"))
    ("i,input", "Self-describing int32 input dump", cxxopts::value<std::string>())
    ("inputangle", "Input map rotation angle (degrees)", cxxopts::value<Tcalc>())
    ("xcenter", "Input center x coordinate", cxxopts::value<Tcalc>())
    ("ycenter", "Input center y coordinate", cxxopts::value<Tcalc>())
    ("inpx", "Input pixel size", cxxopts::value<Tcalc>())
    ("outpx", "Output pixel size", cxxopts::value<Tcalc>())
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

  if (result.count("input")) {
    if (!result.count("output") || !result.count("inputangle") ||
        !result.count("xcenter") || !result.count("ycenter") ||
        !result.count("inpx") || !result.count("outpx"))
      throw std::runtime_error(
          "input mode requires output, inputangle, xcenter, ycenter, inpx, and outpx");

    std::ifstream input(result["input"].as<std::string>(), std::ios::binary);
    int inputWidth;
    int inputHeight;
    double scale;
    input.read(reinterpret_cast<char *>(&inputWidth), sizeof(inputWidth));
    input.read(reinterpret_cast<char *>(&inputHeight), sizeof(inputHeight));
    input.read(reinterpret_cast<char *>(&scale), sizeof(scale));
    if (!input || inputWidth < 1 || inputHeight < 1)
      throw std::runtime_error("invalid input dump header");

    containers::DeviceImageStorage<int> dumpInput(inputHeight, inputWidth);
    thrust::pinned_host_vector<int> hostInput(dumpInput.vec.size());
    input.read(reinterpret_cast<char *>(hostInput.data().get()),
               hostInput.size() * sizeof(int));
    if (!input)
      throw std::runtime_error("input dump payload is truncated");
    dumpInput.vec = hostInput;

    int outputWidth = result["outwidth"].as<int>();
    int outputHeight = result["outheight"].as<int>();
    containers::DeviceImageStorage<int> dumpOutput(outputHeight, outputWidth);
    int factor = result["factor"].as<int>();
    int2 oversampleFactor{factor, factor};
    int minWeight = result["minweight"].as<int>();
    Tcalc inputPixelSize = result["inpx"].as<Tcalc>();
    Tcalc outputPixelSize = result["outpx"].as<Tcalc>();
    cuda_vec2_t<Tcalc> inputCenter{result["xcenter"].as<Tcalc>(),
                                  result["ycenter"].as<Tcalc>()};
    cuda_vec2_t<Tcalc> outputCenter{(outputWidth - 1) * Tcalc(0.5),
                                   (outputHeight - 1) * Tcalc(0.5)};
    cuda_vec2_t<Tcalc> outOffset{
        inputCenter.x - outputCenter.x * outputPixelSize / inputPixelSize,
        inputCenter.y - outputCenter.y * outputPixelSize / inputPixelSize};
    cuda_vec2_t<Tcalc> outStep{outputPixelSize / inputPixelSize,
                               outputPixelSize / inputPixelSize};
    Tcalc angleRadians = -result["inputangle"].as<Tcalc>() / Tcalc(180) * M_PI;
    std::string kernel = result["kernel"].as<std::string>();
    if (kernel == "bilinear") {
      oversampleBilerpAndCombine<int, int, Tcalc, false>(
          dumpInput.cimage(), dumpOutput.image(), oversampleFactor, outOffset,
          outStep, dim3(32, 4), angleRadians, &inputCenter, minWeight);
    } else if (kernel == "bicubic") {
      oversampleBicubicAndCombine<int, int, Tcalc>(
          dumpInput.cimage(), dumpOutput.image(), oversampleFactor, outOffset,
          outStep, dim3(32, 4), angleRadians, &inputCenter, minWeight);
    } else {
      throw std::runtime_error("kernel must be bilinear or bicubic");
    }
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));

    thrust::pinned_host_vector<int> hostOutput = dumpOutput.vec;
    std::ofstream output(result["output"].as<std::string>(), std::ios::binary);
    output.write(reinterpret_cast<char *>(&outputWidth), sizeof(outputWidth));
    output.write(reinterpret_cast<char *>(&outputHeight), sizeof(outputHeight));
    output.write(reinterpret_cast<char *>(&scale), sizeof(scale));
    output.write(reinterpret_cast<char *>(hostOutput.data().get()),
                 hostOutput.size() * sizeof(int));
    if (!output)
      throw std::runtime_error("failed to write output dump");
    printf("Input %dx%d, output %dx%d, offset %.17g %.17g, step %.17g %.17g, angle %.17g degrees\n",
           inputWidth, inputHeight, outputWidth, outputHeight,
           static_cast<double>(outOffset.x), static_cast<double>(outOffset.y),
           static_cast<double>(outStep.x), static_cast<double>(outStep.y),
           static_cast<double>(-result["inputangle"].as<Tcalc>()));
    return 0;
  }

  containers::DeviceImageStorage<int> d_in(result["inheight"].as<int>(),
                                           result["inwidth"].as<int>());
  thrust::pinned_host_vector<int> h_in(d_in.vec.size());
  std::iota(h_in.begin(), h_in.end(), 1);
  d_in.vec = h_in;

  // We use Tcalc for the output as well
  containers::DeviceImageStorage<Tcalc> d_out(result["outheight"].as<int>(),
                                              result["outwidth"].as<int>());
  int2 oversampleFactor{result["factor"].as<int>(), result["factor"].as<int>()};
  int minWeight = result["minweight"].as<int>();
  cuda_vec2_t<Tcalc> outOffset{result["xoffset"].as<Tcalc>(),
                               result["yoffset"].as<Tcalc>()};
  cuda_vec2_t<Tcalc> outStep{result["xstep"].as<Tcalc>(),
                             result["ystep"].as<Tcalc>()};

  Tcalc angleRadians = result["angle"].as<Tcalc>() / 180.0 * M_PI;
  if (useSharedMem) {
    oversampleBilerpAndCombine<int, Tcalc, Tcalc, true>(
        d_in.cimage(), d_out.image(), oversampleFactor, outOffset, outStep,
        dim3(32, 4), angleRadians, nullptr, minWeight);
  } else {
    oversampleBilerpAndCombine<int, Tcalc, Tcalc, false>(
        d_in.cimage(), d_out.image(), oversampleFactor, outOffset, outStep,
        dim3(32, 4), angleRadians, nullptr, minWeight);
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

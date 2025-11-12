#include "wccl.h"
#include "wccl_hybrid.cuh"
#include "wccl_kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>

#include "containers/bitset.cuh"
#include "pinnedalloc.cuh"

#include "cxxopts.hpp"
#include <fstream>
#include <iostream>

// clang-format off
int main(int argc, char* argv[]) {
  // Parse command line args
  cxxopts::Options options("wccl_experiment_cuda", "Cuda experiment of wccl for benchmarks");
  options.add_options()
    ("i,input", "Input file (if not specified, will generate random values). Format is a uint8_t binary map.", cxxopts::value<std::string>())
    ("f,fraction", "Fraction of input to use (for random generated input)", cxxopts::value<double>()->default_value("0.5"))
    ("o,output", "Output file (if output is to be dumped)", cxxopts::value<std::string>())
    ("rows", "Input rows", cxxopts::value<int>()->default_value("8192"))
    ("cols", "Input columns", cxxopts::value<int>()->default_value("1024"))
    ("tilewidth", "Tile width", cxxopts::value<int>()->default_value("32"))
    ("tileheight", "Tile height", cxxopts::value<int>()->default_value("8"))
    ("blockwidth", "Block width", cxxopts::value<int>()->default_value("32"))
    ("blockheight", "Block height", cxxopts::value<int>()->default_value("4"))
    ("windowhdist", "Window horizontal distance (+/-)", cxxopts::value<int>()->default_value("1"))
    ("windowvdist", "Window vertical distance (+/-)", cxxopts::value<int>()->default_value("1"))
    ("h,help", "Print usage")
  ;

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const int rows = result["rows"].as<int>();
  const int cols = result["cols"].as<int>();

  const int tileWidth = result["tilewidth"].as<int>();
  const int tileHeight = result["tileheight"].as<int>();
  const int2 tileDims = {tileWidth, tileHeight};
  printf("Tile dims: (%d, %d)\n", tileDims.x, tileDims.y);

  const int blockWidth = result["blockwidth"].as<int>();
  const int blockHeight = result["blockheight"].as<int>();
  const dim3 tpb(blockWidth, blockHeight);
  printf("Block dims: (%d, %d)\n", blockWidth, blockHeight);

  const int windowHDist = result["windowhdist"].as<int>();
  const int windowVDist = result["windowvdist"].as<int>();
  const int2 windowDist = {windowHDist, windowVDist};
  printf("Window dims: (horz: +/-%d, vert: +/-%d)\n", windowDist.x, windowDist.y);

  std::vector<uint8_t> input(rows*cols);
  if (result.count("input")){
    std::cout << "Using input file: " << result["input"].as<std::string>() << std::endl;
    std::ifstream inputfile(result["input"].as<std::string>().c_str(), std::ios::in | std::ios::binary);
    if (!inputfile.is_open()) {
      std::cerr << "Failed to open input file: " << result["input"].as<std::string>() << std::endl;
      return -1;
    }
    inputfile.read((char*)input.data(), rows*cols);
    inputfile.close();

    // Quickly examine the file
    size_t totalCount = 0;
    for (int i = 0; i < rows*cols; i++) {
      if (input[i]) {
        totalCount++;
      }
    }
    std::cout << "Total marked: " << totalCount << "/" << rows*cols << std::endl;
  }
  else {
    const double fraction = result["fraction"].as<double>();
    std::cout << "Generating random input with fraction: " << fraction << std::endl;
    const int totalMarked = (int)(fraction*rows*cols);
    std::cout << "Total marked: " << totalMarked << "/" << rows*cols << std::endl;
    std::fill(input.begin(), input.begin() + totalMarked, 1);
    std::fill(input.begin() + (int)(fraction*rows*cols), input.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(input.begin(), input.end(), g);

    if (rows <= 64 && cols <= 64) {
      printf("Input:\n%s\n ========================== \n",
             wccl::idxstring<uint8_t>(
             input.data(), rows, cols,
             "%hhu")
             .c_str());
    }
  }
  std::vector<int> activeIdx;
  printf("Filling active index vector...");
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == 1)
      activeIdx.push_back((int)i);
  }
  printf("Done\n");
  std::string outputFilename;
  if (result.count("output")) {
    outputFilename = result["output"].as<std::string>();
  }


  // // ================= Example 1
  // constexpr int rows = 4;
  // constexpr int cols = 5;
  // const std::vector<uint8_t> input = {
  //   1, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   0, 0, 1, 0, 0,
  //   0, 1, 0, 0, 1,
  // };
  // const int2 windowDist = {1, 1};
  // const int2 tileDims = {32, 4};

  // // ================= Example 1a
  // constexpr int rows = 4;
  // constexpr int cols = 5;
  // const std::vector<uint8_t> input = {
  //   0, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   1, 0, 0, 0, 0,
  //   0, 0, 0, 0, 0,
  // };
  // const int2 windowDist = {1, 1};
  // const int2 tileDims = {32, 4};

  // // ================= Example 2
  // constexpr int rows = 8;
  // constexpr int cols = 5;
  //
  // const std::vector<uint8_t> input = {
  //   1, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   0, 0, 1, 0, 0,
  //   0, 1, 0, 0, 1,
  //   0, 0, 0, 0, 1,
  //   0, 1, 0, 1, 0,
  //   1, 0, 0, 0, 0,
  //   0, 0, 0, 0, 0,
  // };
  // const int2 windowDist = {1, 1};
  // const int2 tileDims = {32, 4};

  // // ================= Example 6
  // constexpr int rows = 12;
  // constexpr int cols = 6;
  //
  // const std::vector<uint8_t> input = {
  //   0, 0, 1, 0, 1, 0,
  //   0, 1, 0, 0, 0, 1,
  //   0, 1, 0, 0, 0, 1,
  //   0, 0, 1, 0, 1, 0,
  //   0, 0, 1, 0, 1, 0,
  //   0, 1, 0, 0, 0, 1,
  //   0, 1, 0, 0, 0, 1,
  //   0, 0, 1, 0, 1, 0,
  //   1, 0, 0, 1, 0, 0,
  //   1, 0, 1, 0, 1, 0,
  //   0, 1, 0, 0, 0, 0,
  //   0, 0, 0, 0, 0, 0,
  // };
  // const int2 tileDims = {32, 4};
  // const int2 windowDist = {1, 1};

  typedef short2 KernelColRowType;
  typedef int MappingType;
  typedef uint32_t BitsetType;

  if (input.size() != (size_t)rows * cols) {
    throw std::runtime_error("input size is not equal to rows * cols");
  }
  thrust::host_vector<uint8_t> h_input(rows * cols);
  thrust::copy(input.begin(), input.end(), h_input.begin());
  thrust::device_vector<uint8_t> d_input(rows * cols);
  thrust::copy(h_input.begin(), h_input.end(), d_input.begin());

  thrust::device_vector<MappingType> d_mapping_vec(rows * cols);

  // Encapsulate
  wccl::DeviceImage<uint8_t> d_image(d_input, rows, cols);
  wccl::DeviceImage<MappingType> d_mapping(d_mapping_vec, rows, cols);

#ifdef USE_NEIGHBOURCHAINERHYBRID
  wccl::HybridNeighbourChainer neighbourChainerHybrid(rows, cols);
  neighbourChainerHybrid.fillBeta(h_input.data());
  // DEBUG check of filling
  if (rows <= 64 && cols <= 64) {
    printf("Check of beta after compressing\n");
    printf(wccl::bitstring(&neighbourChainerHybrid.getHostBeta()[0], neighbourChainerHybrid.getRows(), neighbourChainerHybrid.getColElements()).c_str());
  }

  unsigned int seedIdx = neighbourChainerHybrid.getNextBeta();
  while (seedIdx < neighbourChainerHybrid.getRows() * neighbourChainerHybrid.getCols()) {
    neighbourChainerHybrid.execute(seedIdx, windowDist);
    if (rows <= 64 && cols <= 64) {
      printf("Check of beta after execute\n");
      printf(wccl::bitstring(&neighbourChainerHybrid.getHostBeta()[0], neighbourChainerHybrid.getRows(), neighbourChainerHybrid.getColElements()).c_str());
    }

    seedIdx = neighbourChainerHybrid.getNextBeta();
    // break;
  }
  printf("Exited hybrid loop\n");
  if (rows <= 64 && cols <= 64) {
    printf("Check of beta at the end\n");
    printf(wccl::bitstring(&neighbourChainerHybrid.getHostBeta()[0], neighbourChainerHybrid.getRows(), neighbourChainerHybrid.getColElements()).c_str());
  }

#else
  // ========== Kernel 1. Local tile merge (using non atomics)
#ifdef USE_NEIGHBOURCHAINER
  printf("Using neighbour chainer\n");
  // dim3 bpg = wccl::local_chain_neighbours<BitsetType, MappingType>(
  //   d_image, d_mapping, tileDims, windowDist, tpb);
  dim3 bpg = wccl::local_chain_neighbours_v2<BitsetType, MappingType>(
    d_image, d_mapping, tileDims, windowDist, tpb);

#elif USE_ATOMICFREE_LOCAL
  printf("Using atomic-free local merge\n");
  dim3 bpg(d_image.width / tileDims.x + (d_image.width % tileDims.x ? 1 : 0),
           d_image.height / tileDims.y + (d_image.height % tileDims.y ? 1 : 0));
  // NOTE: for now on Linux or at least <CUDA 12.9, putting wrong shared mem
  // size does not seem to trigger compute-sanitizer errors, so be warned!
  size_t shmem = tileDims.x * tileDims.y * 2 * sizeof(KernelColRowType);
  printf("Launching (%d, %d) blks (%d, %d) threads kernel with shmem = %zu\n ",
         bpg.x, bpg.y, tpb.x, tpb.y, shmem);
  wccl::local_connect_kernel<MappingType, KernelColRowType>
      <<<bpg, tpb, shmem>>>(d_image, d_mapping, tileDims, windowDist);
#else
  // ========= Kernel 1. Local tile merge (using atomics)
  #ifdef USE_ACTIVESITES_IN_WINDOW
  dim3 bpg = wccl::local_connect_naive_unionfind<MappingType, true>(d_image, d_mapping, tileDims, windowDist, tpb);
  #else
  dim3 bpg = wccl::local_connect_naive_unionfind<MappingType, false>(d_image, d_mapping, tileDims, windowDist, tpb);
  #endif
#endif
  printf("Blocks per grid = (%d, %d)\n", bpg.x, bpg.y);

  // Pull data and check
  thrust::host_vector<MappingType> h_mapping_vec = d_mapping_vec;

  if (rows <= 64 && cols <= 64) {
    printf("%s\n", wccl::idxstring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, "%2d ", "%2c ")
                       .c_str());

    printf("%s\n", wccl::prettystring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, tileDims.x, tileDims.y)
                       .c_str());
  }

  // Kernel 2. Cross-tile merge
  thrust::pinned_host_vector<unsigned int> h_counter(1);
  h_counter[0] = 0;
  thrust::device_vector<unsigned int> d_counter(1);
  thrust::copy(h_counter.begin(), h_counter.end(), d_counter.begin());

  size_t numUnionFindIters = wccl::naive_global_unionfind<MappingType>(
    d_mapping, tileDims, windowDist, tpb, h_counter, d_counter);

  printf("numUnionFindIters = %zu\n", numUnionFindIters);
  h_mapping_vec = d_mapping_vec;

  if (rows <= 64 && cols <= 64) {
    printf("%s\n ========================== \n",
           wccl::idxstring<MappingType>(
               thrust::raw_pointer_cast(h_mapping_vec.data()), rows, cols,
               "%2d ", "%2c ")
               .c_str());
    printf("%s\n", wccl::prettystring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, tileDims.x, tileDims.y)
                       .c_str());
  }

  // Kernel 3a. Path compression (or readout)
  wccl::naive_global_pathcompress<MappingType>(d_mapping, tpb);
  h_mapping_vec = d_mapping_vec;

  if (rows <= 64 && cols <= 64) {
    printf("%s\n ========================== \n",
           wccl::idxstring<MappingType>(
               thrust::raw_pointer_cast(h_mapping_vec.data()), rows, cols,
               "%2d ", "%2c ")
               .c_str());
    printf("%s\n", wccl::prettystring<MappingType>(
                       thrust::raw_pointer_cast(h_mapping_vec.data()), rows,
                       cols, tileDims.x, tileDims.y)
                       .c_str());
  }

  // Kernel 3b. Alternative readout via (row, col, label) triplets
  size_t maxCount = d_mapping.size();
  thrust::device_vector<short2> d_clusterlistpos(maxCount);
  thrust::device_vector<int> d_clusterlistlabel(maxCount);
  thrust::device_vector<unsigned int> d_clusterlistlength(1);
  thrust::pinned_host_vector<unsigned int> h_clusterlistlength(1);
  wccl::naive_global_readout<MappingType>(d_mapping, d_clusterlistpos, d_clusterlistlabel, d_clusterlistlength, tpb, true, &h_clusterlistlength);
  thrust::pinned_host_vector<short2> h_clusterlistpos = d_clusterlistpos;
  thrust::pinned_host_vector<int> h_clusterlistlabel = d_clusterlistlabel;
  if (rows <= 64 && cols <= 64) {
    for (size_t i = 0; i < h_clusterlistlength[0]; ++i) {
      auto& h_clusterpixelpos = h_clusterlistpos[i];
      auto& h_clusterpixellabel = h_clusterlistlabel[i];
      printf("%zu: (%hd, %hd -> %d)\n", i, h_clusterpixelpos.x, h_clusterpixelpos.y,
             h_clusterpixellabel);
    }
  }

  thrust::device_vector<int> d_unique_labels(maxCount);
  // thrust::device_vector<unsigned int> d_label_counts(maxCount);
  thrust::device_vector<wccl::ClusterStatistics<short2>> d_stats(maxCount);
  unsigned int numUniqueLabels = wccl::calculateClusterStatistics(
    d_clusterlistpos, d_clusterlistlabel, d_unique_labels,
    // d_label_counts,
    d_stats,
    h_clusterlistlength[0]
  );

  thrust::host_vector<int> h_unique_labels = d_unique_labels;
  thrust::host_vector<wccl::ClusterStatistics<short2>> h_stats(h_clusterlistlength[0]);
  thrust::copy(d_stats.begin(), d_stats.begin() + h_clusterlistlength[0], h_stats.begin());
  for (unsigned int i = 0; i < numUniqueLabels; ++i){
    auto& h_cluster = h_stats[i];
    printf("Cluster %u: min (%d, %d), max (%d, %d)\n",
           i, h_cluster.min.x, h_cluster.min.y, h_cluster.max.x, h_cluster.max.y);
  }
  printf("Clusterlist length = %u / %zu\n", h_clusterlistlength[0], maxCount);
#endif

  return 0;
}
// clang-format on

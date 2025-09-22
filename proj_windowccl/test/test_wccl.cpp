#include <gtest/gtest.h>

#include "wccl.h"
#include "wccl_kernels.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define ERROR_EXPECTED_INACTIVE_GOT_ACTIVE 1
#define ERROR_EXPECTED_ACTIVE_GOT_INACTIVE 2
#define ERROR_NEIGHBOUR_MISMATCH           3

#define METHOD_LOCAL_NAIVE 0
#define METHOD_LOCAL_NAIVE_ATOMICFREE 1
#define METHOD_LOCAL_NEIGHBOURCHAIN 2

template <typename T>
int validatePointBasic(
  const uint8_t* input,
  const T* mapping,
  const int height,
  const int width,
  const int hDist,
  const int vDist,
  const int i,
  const int j
){
  T centre = mapping[i * width + j];
  uint8_t inputVal = input[i * width + j];
  // Mapping (output) point is inactive
  if (centre < 0)
  {
    // Input point is also inactive, correct
    if (inputVal == 0)
      return 0;
    else
      return ERROR_EXPECTED_ACTIVE_GOT_INACTIVE; // output is wrongly classified as inactive
  }
  // Mapping point is active
  else{
    // Input point is inactive, wrong
    if (inputVal == 0)
      return ERROR_EXPECTED_INACTIVE_GOT_ACTIVE; // output is wrongly classified as active
  }

  // If active, check all neighbours
  for (int y = i - vDist; y <= i + vDist; ++y){
    if (y < 0 || y >= height)
      continue;
    for (int x = j - hDist; x <= j + hDist; ++x){
      if (x < 0 || x >= width)
        continue;
      // Read neighbour
      T neighbourValue = mapping[y * width + x];
      // Ignore inactive neighbours
      if (neighbourValue < 0)
        continue;
      // Active neighbours must match
      if (mapping[y * width + x] != centre)
        return ERROR_NEIGHBOUR_MISMATCH;
    }
  }
  // All pass then true
  return 0;
}

template <typename T>
int validatePoint(
  const std::vector<uint8_t>& input,
  const wccl::CPUMapping<T>& mapping,
  const int hDist,
  const int vDist,
  const int i,
  const int j
){
  return validatePointBasic(input.data(), mapping.data.data(), (int)mapping.height, (int)mapping.width, hDist, vDist, i, j);
}

template <typename T>
void validate(const std::vector<uint8_t>& input, const wccl::CPUMapping<T>& mapping, const int hDist, const int vDist){
  for (int i = 0; i < (int)mapping.height; ++i){
    for (int j = 0; j < (int)mapping.width; ++j){
      if (mapping.data[i * mapping.width + j] < 0)
        continue;
      EXPECT_EQ(validatePoint(input, mapping, hDist, vDist, i, j), 0) << mapping.tostring();
    }
  }
}

template <typename T>
void copyTile(const T* src, const int srcWidth, const int srcHeight, T* dst, const int dstWidth, const int dstHeight, const int startRow, const int startCol, const T invalidValue){
  for (int i = 0; i < dstHeight; ++i){
    for (int j = 0; j < dstWidth; ++j){
      int srow = i + startRow;
      int scol = j + startCol;
      if (srow < 0 || srow >= srcHeight || scol < 0 || scol >= srcWidth)
        dst[i * dstWidth + j] = invalidValue;
      else
        dst[i * dstWidth + j] = src[srow * srcWidth + scol];
    }
  }
}

template <typename Tmapping, int method = METHOD_LOCAL_NAIVE, typename Tbitset = unsigned int>
void localTileCudaTest(
  const std::vector<uint8_t>& img,
  const int rows,
  const int cols,
  const dim3 tpb,
  const int2 tileDims,
  const int2 windowDist
){
  thrust::host_vector<uint8_t> h_imgvec(img.size());
  thrust::copy(img.begin(), img.end(), h_imgvec.begin());
  thrust::device_vector<uint8_t> d_imgvec = h_imgvec;

  thrust::device_vector<Tmapping> d_mappingvec(rows * cols);
  
  wccl::DeviceImage<uint8_t> d_img(d_imgvec, rows, cols);
  wccl::DeviceImage<Tmapping> d_mapping(d_mappingvec, rows, cols);

  dim3 bpg;
  static_assert(method == METHOD_LOCAL_NAIVE || method == METHOD_LOCAL_NEIGHBOURCHAIN,
    "Method must be either METHOD_LOCAL_NAIVE or METHOD_LOCAL_NEIGHBOURCHAIN (for now)");
  if constexpr(method == METHOD_LOCAL_NAIVE){
    bpg = wccl::local_connect_naive_unionfind<Tmapping>(d_img, d_mapping, tileDims, windowDist, tpb);
  }
  else if constexpr(method == METHOD_LOCAL_NEIGHBOURCHAIN){
    bpg = wccl::local_chain_neighbours<Tbitset, Tmapping>(d_img, d_mapping, tileDims, windowDist, tpb);
  }

  thrust::host_vector<Tmapping> h_mappingvec = d_mappingvec;

  // Read each tile separately
  for (int i = 0; i < (int)bpg.y; ++i){
    for (int j = 0; j < (int)bpg.x; ++j){
      int startRow = i * tileDims.y;
      int startCol = j * tileDims.x;
      std::vector<uint8_t> tileinputvec(tileDims.x * tileDims.y);
      copyTile<uint8_t>(h_imgvec.data(), cols, rows, tileinputvec.data(), tileDims.x, tileDims.y, startRow, startCol, 0);
      std::vector<Tmapping> tilemappingvec(tileDims.x * tileDims.y);
      copyTile<Tmapping>(h_mappingvec.data(), cols, rows, tilemappingvec.data(), tileDims.x, tileDims.y, startRow, startCol, -1);

      std::string tilemappingstr;
      char tmp[8];
      if (tileDims.x <= 64 && tileDims.y <= 64){
        for (int ti = 0; ti < (int)tileDims.y; ++ti){
          for (int tj = 0; tj < (int)tileDims.x; ++tj){
            snprintf(tmp, 8, "%2d", tilemappingvec[ti * tileDims.x + tj]);
            tilemappingstr += std::string(tmp) + " ";
          }
          tilemappingstr += "\n";
        }
      }

      std::string tileinputstr;
      for (int ti = 0; ti < (int)tileDims.y; ++ti){
        for (int tj = 0; tj < (int)tileDims.x; ++tj){
          snprintf(tmp, 8, "%2d", tileinputvec[ti * tileDims.x + tj]);
          tileinputstr += std::string(tmp) + " ";
        }
        tileinputstr += "\n";
      }

      for (int ii = 0; ii < (int)tileDims.y; ++ii){
        for (int jj = 0; jj < (int)tileDims.x; ++jj){
          char errmsg[2048];
          snprintf(errmsg, sizeof(errmsg),
                   "tile (%d,%d), idx (%d,%d), coords (%d,%d) = %d // input is %hhu\n",
                   i, j, ii, jj, i*tileDims.y + ii, j*tileDims.x + jj, tilemappingvec[ii * tileDims.x + jj], tileinputvec[ii * tileDims.x + jj]);
          ASSERT_EQ(validatePointBasic(
            tileinputvec.data(), tilemappingvec.data(), tileDims.y, tileDims.x, windowDist.x, windowDist.y, ii, jj
          ), 0) << errmsg + tilemappingstr + "\n-------------------------------------------------------------\n" + tileinputstr;
        }
      }
    }
  }
}


// ======================================================================
// ======================================================================
// ========================= TESTS ======================================
// ======================================================================
// ======================================================================

TEST(WindowCCL, CPUSolver_basic) {
  uint32_t height = 5, width = 5;
  int hDist = 1, vDist = 1;

  wccl::CPUMapping mapping(height, width);
  wccl::CPUSolver solver(hDist, vDist);

  std::vector<uint8_t> img = {
    0, 0, 0, 0, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    0, 0, 1, 0, 0
  };

  solver.connect(img, mapping.height, mapping.width, mapping.data);
  solver.readout(height, width, mapping.data);

  validate(img, mapping, hDist, vDist);
}

TEST(CudaWindowCCL, NaiveLocal_basic1){
  std::vector<uint8_t> img = {
    0, 0, 0, 0, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    0, 0, 1, 0, 0
  };
  int rows = 5, cols = 5;
  dim3 tpb(32,4);
  int2 tileDims = {32, 4};
  int2 windowDist = {1, 1};
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NaiveLocal_basic2){
  constexpr int rows = 4;
  constexpr int cols = 5;
  const std::vector<uint8_t> img = {
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 0, 1,
  };
  const int2 windowDist = {1, 1};
  const int2 tileDims = {32, 4};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}


TEST(CudaWindowCCL, NaiveLocal_basic3){
  constexpr int rows = 4;
  constexpr int cols = 5;
  const std::vector<uint8_t> img = {
    0, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
  };
  const int2 windowDist = {1, 1};
  const int2 tileDims = {32, 4};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NaiveLocal_basic4){
  constexpr int rows = 8;
  constexpr int cols = 5;
  const std::vector<uint8_t> img = {
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 0, 1,
    0, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
  };
  const int2 windowDist = {1, 1};
  const int2 tileDims = {32, 4};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NaiveLocal_basic5){
  constexpr int rows = 12;
  constexpr int cols = 6;

  const std::vector<uint8_t> img = {
    0, 0, 1, 0, 1, 0,
    0, 1, 0, 0, 0, 1,
    0, 1, 0, 0, 0, 1,
    0, 0, 1, 0, 1, 0,
    0, 0, 1, 0, 1, 0,
    0, 1, 0, 0, 0, 1,
    0, 1, 0, 0, 0, 1,
    0, 0, 1, 0, 1, 0,
    1, 0, 0, 1, 0, 0,
    1, 0, 1, 0, 1, 0,
    0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
  };
  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NaiveLocal_random64x64_1percent){
  constexpr int rows = 64;
  constexpr int cols = 64;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.01;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}


TEST(CudaWindowCCL, NaiveLocal_random8192x1024_1percent){
  constexpr int rows = 8192;
  constexpr int cols = 1024;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.01;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NaiveLocal_random8192x1024_50percent){
  constexpr int rows = 8192;
  constexpr int cols = 1024;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.50;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NeighbourChainerLocal_custom1){
  constexpr int rows = 4;
  constexpr int cols = 32;

  const std::vector<uint8_t> img = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
  };
  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NeighbourChainerLocal_random64x64_1percent){
  constexpr int rows = 64;
  constexpr int cols = 64;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.01;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int, METHOD_LOCAL_NEIGHBOURCHAIN>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NeighbourChainerLocal_random64x64_50percent){
  constexpr int rows = 64;
  constexpr int cols = 64;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.5;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int, METHOD_LOCAL_NEIGHBOURCHAIN>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NeighbourChainerLocal_random512x512_1percent){
  constexpr int rows = 512;
  constexpr int cols = 512;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.01;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int, METHOD_LOCAL_NEIGHBOURCHAIN>(img, rows, cols, tpb, tileDims, windowDist);
}


TEST(CudaWindowCCL, NeighbourChainerLocal_random8192x1024_1percent){
  constexpr int rows = 8192;
  constexpr int cols = 1024;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.01;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int, METHOD_LOCAL_NEIGHBOURCHAIN>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NeighbourChainerLocal_random8192x1024_50percent){
  constexpr int rows = 8192;
  constexpr int cols = 1024;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.50;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int, METHOD_LOCAL_NEIGHBOURCHAIN>(img, rows, cols, tpb, tileDims, windowDist);
}

TEST(CudaWindowCCL, NeighbourChainerLocal_uchar_random8192x1024_50percent){
  constexpr int rows = 8192;
  constexpr int cols = 1024;

  std::vector<uint8_t> img(rows * cols);
  const double fraction = 0.50;
  std::fill(img.begin(), img.begin() + (int)(fraction * rows * cols), 1);
  std::fill(img.begin() + (int)(fraction * rows * cols), img.end(), 0);
  std::random_shuffle(img.begin(), img.end());

  const int2 tileDims = {32, 4};
  const int2 windowDist = {1, 1};
  dim3 tpb(32,4);
  localTileCudaTest<int, METHOD_LOCAL_NEIGHBOURCHAIN, unsigned char>(img, rows, cols, tpb, tileDims, windowDist);
}

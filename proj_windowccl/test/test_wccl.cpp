#include <gtest/gtest.h>

#include "wccl.h"
#include "wccl_kernels.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename T>
bool validatePointBasic(
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
  T inputVal = input[i * width + j];
  // Check that if input is inactive then this should be nactive
  if (centre < 0)
  {
    if (inputVal == 0)
      return true;
    else
      return false; // output is wrongly classified as inactive
  }
  else{
    if (inputVal == 0)
      return false; // output is wrongly classified as active
  }

  // If active, check all neighbours
  for (int x = i - hDist; x <= i + hDist; ++x){
    for (int y = j - vDist; y <= j + vDist; ++y){
      if (x < 0 || x >= height || y < 0 || y >= width)
        continue;
      if (mapping[y * width + x] >= 0 && mapping[y * width + x] != centre)
        return false;
    }
  }
  // All pass then true
  return true;
}

template <typename T>
bool validatePoint(
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
      EXPECT_TRUE(validatePoint(input, mapping, hDist, vDist, i, j)) << mapping.tostring();
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

template <typename T>
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

  thrust::device_vector<T> d_mappingvec(rows * cols);
  
  wccl::DeviceImage<uint8_t> d_img(d_imgvec, rows, cols);
  wccl::DeviceImage<T> d_mapping(d_mappingvec, rows, cols);

  dim3 bpg = wccl::local_connect_naive_unionfind<T>(d_img, d_mapping, tileDims, windowDist, tpb);

  thrust::host_vector<T> h_mappingvec = d_mappingvec;

  // Read each tile separately
  for (int i = 0; i < (int)bpg.y; ++i){
    for (int j = 0; j < (int)bpg.x; ++j){
      int startRow = i * tileDims.y;
      int startCol = j * tileDims.x;
      std::vector<uint8_t> tileinputvec(tileDims.x * tileDims.y);
      copyTile<uint8_t>(h_imgvec.data(), cols, rows, tileinputvec.data(), tileDims.x, tileDims.y, startRow, startCol, 0);
      std::vector<T> tilemappingvec(tileDims.x * tileDims.y);
      copyTile<T>(h_mappingvec.data(), cols, rows, tilemappingvec.data(), tileDims.x, tileDims.y, startRow, startCol, -1);

      for (int ii = 0; ii < (int)tileDims.y; ++ii){
        for (int jj = 0; jj < (int)tileDims.x; ++jj){
          EXPECT_TRUE(validatePointBasic(
            tileinputvec.data(), tilemappingvec.data(), rows, cols, windowDist.x, windowDist.y, ii, jj
          ));
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

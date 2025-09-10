#include <gtest/gtest.h>

#include "wccl.h"

template <typename T>
bool validatePoint(
  const wccl::CPUMapping<T>& mapping,
  const int hDist,
  const int vDist,
  const int i,
  const int j
){
  T centre = mapping.data[i * mapping.width + j];
  if (centre < 0)
    return true;

  for (int x = i - hDist; x <= i + hDist; ++x){
    for (int y = j - vDist; y <= j + vDist; ++y){
      if (x < 0 || x >= (int)mapping.height || y < 0 || y >= (int)mapping.width)
        continue;
      if (mapping.at(y, x) >= 0 && mapping.at(y, x) != centre)
        return false;
    }
  }
  return true;
}

template <typename T>
void validate(const wccl::CPUMapping<T>& mapping, const int hDist, const int vDist){
  for (int i = 0; i < (int)mapping.height; ++i){
    for (int j = 0; j < (int)mapping.width; ++j){
      if (mapping.data[i * mapping.width + j] < 0)
        continue;
      EXPECT_TRUE(validatePoint(mapping, hDist, vDist, i, j)) << mapping.tostring();
    }
  }
}


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

  validate(mapping, hDist, vDist);
}

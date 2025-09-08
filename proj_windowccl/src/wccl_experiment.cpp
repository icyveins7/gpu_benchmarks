#include "wccl.h"
#include <iostream>

int main()
{
  printf("wccl experiments\n");

  uint32_t height = 3, width = 5;
  wccl::CPUMapping mapping(height, width);
  wccl::CPUSolver solver(1,1);

  std::vector<uint8_t> img = {
    0, 0, 1, 0, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 1
  };

  solver.connect(img, mapping.height, mapping.width, mapping.data);

  mapping.print();

  return 0;
}

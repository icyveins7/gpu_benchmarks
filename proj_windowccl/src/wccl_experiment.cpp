#include "wccl.h"
#include <iostream>
#include <cstdlib>
#include "timer.h"

void experiment(
  const uint32_t height,
  const uint32_t width,
  const int hDist,
  const int vDist,
  const std::vector<uint8_t>& img,
  const bool print = true
){
  printf("Experiment: height = %u, width = %u, hDist = %d, vDist = %d\n", height, width, hDist, vDist);
  wccl::CPUMapping mapping(height, width);
  wccl::CPUSolver solver(hDist, vDist);

  {
    HighResolutionTimer timer;
    solver.connect(img, mapping.height, mapping.width, mapping.data);
    solver.readout(height, width, mapping.data);
  }

  if (print){
    mapping.print();
  }

  printf("=========\n\n");
}

int main()
{
  printf("wccl experiments\n");

  // Testing hDist = 1, vDist = 1
  {
    int hDist = 1, vDist = 1;

    {
      uint32_t height = 3, width = 5;
      std::vector<uint8_t> img = {
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1
      };
      experiment(height, width, hDist, vDist, img);
    }
    {
      uint32_t height = 5, width = 5;
      std::vector<uint8_t> img = {
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0
      };
      experiment(height, width, hDist, vDist, img);
    }
    {
      uint32_t height = 5, width = 5;
      std::vector<uint8_t> img = {
        0, 0, 0, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0
      };
      experiment(height, width, hDist, vDist, img);
    }
    {
      uint32_t height = 5, width = 5;
      std::vector<uint8_t> img = {
        0, 0, 0, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 0, 0, 0
      };
      experiment(height, width, hDist, vDist, img);
    }
    {
      uint32_t height = 5, width = 5;
      std::vector<uint8_t> img = {
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0
      };
      experiment(height, width, hDist, vDist, img);
    }
    {
      uint32_t height = 8192, width = 1024;
      std::vector<uint8_t> img(height * width);
      for (size_t i = 0; i < height * width; ++i)
        img.at(i) = rand() % 2;
      experiment(height, width, hDist, vDist, img, false);
    }
  } // end hDist = 1, vDist = 1

  {
    int hDist = 1, vDist = 2;

    {
      uint32_t height = 5, width = 5;
      std::vector<uint8_t> img = {
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0
      };
      experiment(height, width, hDist, vDist, img);
    }
    {
      uint32_t height = 5, width = 5;
      std::vector<uint8_t> img = {
        1, 0, 1, 0, 1,
        0, 0, 0, 0, 0,
        0, 1, 0, 0, 1,
        0, 0, 0, 0, 0,
        1, 0, 1, 0, 1
      };
      experiment(height, width, hDist, vDist, img);
    }
  } // end hDist = 1, vDist = 2


  return 0;
}

#include "wccl.h"

#include <stdexcept>
#include <algorithm>

namespace wccl
{

CPUSolver::CPUSolver(const uint32_t hDist, const uint32_t vDist)
  : m_hDist(hDist), m_vDist(vDist)
{
}

void CPUSolver::connect(const uint8_t* img, const uint32_t height, const uint32_t width, int32_t* mappings)
{
  // 1. Initialize all active sites
  for (uint32_t i = 0; i < height; i++)
  {
    for (uint32_t j = 0; j < width; j++)
    {
      // Non-active sites are marked with -1
      if (img[i * width + j] == 0)
        mappings[i * width + j] = -1;
      else
        mappings[i * width + j] = i * width + j;
    }
  }

  // 2. Unite within windows and flatten (path compression)
  for (int32_t i = 0; i < (int32_t)height; i++){
    for (int32_t j = 0; j < (int32_t)width; j++){
      // Ignore inactive sites
      if (mappings[i * width + j] == -1)
        continue;

      // Get root of current pixel
      int32_t root = find(mappings, i, j, width);

      // Search within the window
      for (int32_t wi = i - m_vDist; wi <= i + (int32_t)m_vDist; wi++){
        // Ignore if out of range
        if (wi < 0 || wi >= (int32_t)height)
          continue;
        for (int32_t wj = j - m_hDist; wj <= j + (int32_t)m_hDist; wj++){
          // Ignore if out of range
          if (wj < 0 || wj >= (int32_t)width)
            continue;

          // Ignore if not active
          if (mappings[wi * width + wj] == -1)
            continue;

          // Otherwise, follow the path to root for candidate pixel
          int32_t candroot = find(mappings, wi, wj, width);

          // Unite if not already connected
          if (root != candroot){
            int32_t unitedroot = std::min(root, candroot);

            // Path compress the current pixel
            if (root != unitedroot)
              pathcompress(mappings, i, j, width, unitedroot);
            // Otherwise path compress the candidate pixel
            else
              pathcompress(mappings, wi, wj, width, unitedroot);
          }
        } // end window loops (columns)
      } // end window loops (rows)
    } // end pixel loops (columns)
  } // end pixel loops (rows)


}

void CPUSolver::connect(const std::vector<uint8_t>& img, const uint32_t height, const uint32_t width, std::vector<int32_t>& mappings){
  if (img.size() != (size_t)height * width)
    throw std::runtime_error("Invalid input image size");
  if (mappings.size() != img.size())
    throw std::runtime_error("Invalid mappings size; should equal input img size");

  this->connect(img.data(), height, width, mappings.data());
}

void CPUSolver::readout(const uint32_t height, const uint32_t width, std::vector<int32_t>& mapping){
  for (uint32_t i = 0; i < height; ++i){
    for (uint32_t j = 0; j < width; ++j){
      int32_t root = this->find(mapping.data(), i, j, width);
      mapping[i * width + j] = root;
    }
  }
}

int32_t CPUSolver::find(int32_t* mappings, const uint32_t i, const uint32_t j, const uint32_t width){
  if (mappings[i * width + j] < 0)
    return -1;

  int32_t root = mappings[i * width + j];
  while (root != mappings[root])
    root = mappings[root];

  return root;
}

void CPUSolver::pathcompress(int32_t* mappings, const uint32_t i, const uint32_t j, const uint32_t width, const int32_t root){
  int32_t* currentPtr = &mappings[i * width + j];
  int32_t currentParent = *currentPtr;
  while (currentParent != root){
    // Change the current pointer value
    *currentPtr = root;
    // Go to the parent
    currentPtr = &mappings[currentParent];
    // Update the parent
    currentParent = *currentPtr;
  }
}

}

#pragma once

#include <cstdint>
#include <vector>

/*
All code lies inside our [W]indow-[C]onnected-[C]omponent-[L]abelling (WCCL) namespace.
The method is not really standard CCL, nor is it completely identical to the disjoint-set union-find.
I just like how wccl sounds.
*/
namespace wccl
{

struct CPUMapping
{
  explicit CPUMapping(const uint32_t _height, const uint32_t _width);
  std::vector<int32_t> data;
  uint32_t height;
  uint32_t width;

  void print(const char* fmt="%d ");
};

class CPUSolver
{
public:
  /**
   * @brief Constructs a new CPUSolver, which can be used to cluster input pixels based on a window.
   *
   * @param hDist Maximum horizontal distance to cluster i.e. x <= x + hDist
   * @param vDist Maximum vertical distance to cluster i.e. y <= y + vDist
   */
  explicit CPUSolver(const uint32_t hDist, const uint32_t vDist);

  /**
   * @brief Primary method to cluster input pixels.
   * @detail This base class implements the union-find method as a baseline.
   *
   * @param img Input image
   * @param height Input image height
   * @param width Input image width
   * @param mappings Output mapping, same dimensions as input image
   */
  virtual void connect(const uint8_t* img, const uint32_t height, const uint32_t width, int32_t* mappings);

  /**
   * @brief Wrapper for the C-array implementation method
   */
  void connect(const std::vector<uint8_t>& img, const uint32_t height, const uint32_t width, std::vector<int32_t>& mappings);

protected:
  uint32_t m_hDist;
  uint32_t m_vDist;

private:
  int32_t find(int32_t* mappings, const uint32_t i, const uint32_t j, const uint32_t width);
  void pathcompress(int32_t* mappings, const uint32_t i, const uint32_t j, const uint32_t width, const int32_t root);
};

} // end namespace wccl

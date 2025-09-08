#pragma once
/*
All code lies inside our [W]indow-[C]onnected-[C]omponent-[L]abelling (WCCL) namespace.
The method is not really standard CCL, nor is it completely identical to the disjoint-set union-find.
I just like how wccl sounds.
*/
#include <cstdint>
namespace wccl
{

class CPUSolver
{
public:
  explicit CPUSolver(const int hDist, const int vDist);

  void connect(const uint8_t* img, const int height, const int width, int32_t mappings);

private:
  int m_hDist;
  int m_vDist;
};

} // end namespace wccl

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

/*
All code lies inside our [W]indow-[C]onnected-[C]omponent-[L]abelling (WCCL) namespace.
The method is not really standard CCL, nor is it completely identical to the disjoint-set union-find.
I just like how wccl sounds.
*/
namespace wccl
{

template <typename T = int32_t>
struct CPUMapping
{
  std::vector<T> data;
  uint32_t height;
  uint32_t width;

  explicit CPUMapping(const uint32_t _height, const uint32_t _width)
    : data(_height * _width), height(_height), width(_width){};

  // Only support const-getters
  const T& at(const uint32_t i, const uint32_t j) const{
    if (i >= height || j >= width)
      throw std::out_of_range("Index out of range");
    return data.at(i * width + j);
  }

  std::string tostring(const char* fmt="%d ", const bool prettify = true) const{
    std::vector<char> labelChars;
      std::unordered_map<int32_t, char> labelMap;
    if (prettify){
      // A-Z
      for (uint8_t i = 65; i < 91; ++i)
        labelChars.push_back(i);

      // 0-9
      for (uint8_t i = 48; i < 58; ++i)
        labelChars.push_back(i);

      size_t ctr = 0;
      for (size_t i = 0; i < data.size(); ++i){
        if (data.at(i) >= 0 && labelMap.find(data.at(i)) == labelMap.end()){
          labelMap.insert({data.at(i), labelChars.at(ctr)});
          ctr++;
        }
      }
    }

    std::string s;
    char temp[8];
    for (uint32_t i = 0; i < height; ++i){
      for (uint32_t j = 0; j < width; ++j){
        if (data[i * width + j] == -1)
          s += "- ";
        else
        {
          if (prettify)
            snprintf(temp, 8, "%c ", labelMap.at(data[i * width + j]));
          else 
            // Print the value itself
            snprintf(temp, 8, fmt, data[i * width + j]);

          s += temp;
        }
      }
      s += "\n";
    }
    return s;
  }

  void print(const char* fmt = "%d ", const bool prettify = true) const{
    printf(this->tostring(fmt, prettify).c_str());
  }
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

  /**
   * @brief Performs a final read out of the mapping by following all elements to their roots,
   * effectively labelling each element.
   *
   * @param height Input image height
   * @param width Input image width
   * @param mapping Output mapping
   */
  void readout(const uint32_t height, const uint32_t width, std::vector<int32_t>& mapping);

protected:
  uint32_t m_hDist;
  uint32_t m_vDist;

private:
  int32_t find(int32_t* mappings, const uint32_t i, const uint32_t j, const uint32_t width);
  void pathcompress(int32_t* mappings, const uint32_t i, const uint32_t j, const uint32_t width, const int32_t root);
};

} // end namespace wccl

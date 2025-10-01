#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdexcept>

namespace containers {

/**
 * @brief A container that represents a view over some data containing raw bits.
 */
template <typename T = unsigned int, typename U = int> struct BitsetArray {
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");

  T *data = nullptr;
  U numDataElements = 0;
  U numBits = 0;

  __host__ __device__ BitsetArray() {};
  __host__ __device__ BitsetArray(T *_data, U _numDataElements, U _numBits)
      : data(_data), numDataElements(_numDataElements), numBits(_numBits) {
    // There is no easy way to check this inside the kernel since we cannot
    // throw, so it is expected that the numBits can fit inside numDataElements
  }
  __host__ __device__ BitsetArray(T *_data, U _numBits)
      : data(_data), numBits(_numBits) {
    // If only bits provided, assume the data elements were allocated using
    // numElementsRequiredFor().
    numDataElements = this->numElementsRequiredFor(_numBits);
  }
  __host__ BitsetArray(thrust::device_vector<T> &_data, U _numBits)
      : data(_data.data().get()), numDataElements(_data.size()),
        numBits(_numBits) {
    if (numBits > numDataElements * numBitsPerElement()) {
      throw std::invalid_argument(
          "numBits > numDataElements * numBitsPerElement()");
    }
  }
  __host__ BitsetArray(thrust::host_vector<T> &_data, U _numBits)
      : data(_data.data()), numDataElements(_data.size()), numBits(_numBits) {
    if (numBits > numDataElements * numBitsPerElement()) {
      throw std::invalid_argument(
          "numBits > numDataElements * numBitsPerElement()");
    }
  }

  /**
   * @brief Returns the number of elements required for a given number of bits,
   * for the current templated type. Useful to help in allocation size for the
   * templated type.
   *
   * @param numBits Total number of desired bits
   * @return Total number of required elements of type T
   */
  __host__ __device__ static U numElementsRequiredFor(const U numBits) {
    U numElements = numBits / numBitsPerElement();
    if (numBits % numBitsPerElement() > 0)
      numElements++;
    return numElements;
  }

  __host__ __device__ T maskElement(const T element, const U offset) const {
    return element & (1 << offset);
  }

  __host__ __device__ static constexpr U numBitsPerElement() {
    return sizeof(T) * 8;
  }

  __host__ __device__ bool hasValidNumBits() const {
    return numBits <= numDataElements * numBitsPerElement();
  }

  // ===================================================================
  // ====================== ELEMENT MANIPULATION =======================
  // ===================================================================

  __host__ __device__ bool isValidElementIndex(const U index) const {
    return index < numDataElements;
  }

  __host__ __device__ T &elementAt(const U index) { return data[index]; }
  __host__ __device__ const T &elementAt(const U index) const {
    return data[index];
  }

  /**
   * @brief Returns the element containing the bit index. See bitOffset() to
   * retrieve the bit offset within the element.
   *
   * @param bIndex Bit index
   * @return Element reference
   */
  __host__ __device__ T &elementContainingBitAt(const U bIndex) {
    return elementAt(bIndex / numBitsPerElement());
  }

  /**
   * @brief Returns the element containing the bit index. See bitOffset() to
   * retrieve the bit offset within the element.
   *
   * @param bIndex Bit index
   * @return Const element reference
   */
  __host__ __device__ const T &elementContainingBitAt(const U bIndex) const {
    return elementAt(bIndex / numBitsPerElement());
  }

  // ===================================================================
  // ========================== BIT MANIPULATION =======================
  // ===================================================================

  /**
   * @brief Returns the element index containing the bit at the specified bit
   * index.
   *
   * @param bIndex Bit index
   * @return Element index
   */
  __host__ __device__ U elementOffset(const U bIndex) const {
    return bIndex / numBitsPerElement();
  }

  /**
   * @brief Returns the number of bits offset inside the element containing the
   bit index. See elementContainingBitAt() to retrieve the element itself.

   * @param bIndex Bit index
   * @return Number of bits offset
   */
  __host__ __device__ U bitOffset(const U bIndex) const {
    return bIndex % numBitsPerElement();
  }

  __host__ __device__ bool isValidBitIndex(const U bIndex) const {
    return bIndex < numBits;
  }

  /**
   * @brief Primary getter for bits.
   *
   * @param bIndex Bit index
   * @return True if the bit is set
   */
  __host__ __device__ bool getBitAt(const U bIndex) const {
    const U offset = bitOffset(bIndex);
    const T element = elementContainingBitAt(bIndex);
    return (maskElement(element, bIndex % numBitsPerElement()) >> offset) == 1;
  }

  /**
   * @brief Primary setter for bits.
   *
   * @param bIndex Bit index
   * @param value Value to set. Using truthy-values like 1 or 0 directly also
   * works.
   */
  __host__ __device__ void setBitAt(const U bIndex, const bool value) {
    const U offset = bitOffset(bIndex);
    T &element = elementContainingBitAt(bIndex);
    if (value) // set bit to 1
      element |= (1 << offset);
    else // set bit to 0
      element &= ~(1 << offset);
  }

  /**
   * @brief Setter for bits atomically. Use this if multiple threads will write
   * to the same element.
   *
   * @param bIndex Bit index
   * @param value Value to set. Using truthy-values like 1 or 0 directly also
   * works.
   */
  __device__ void setBitAtAtomically(const U bIndex, const bool value) {
    const U offset = bitOffset(bIndex);
    T &element = elementContainingBitAt(bIndex);
    if (value) // set bit to 1
      atomicOr(&element, (1 << offset));
    else // set bit to 0
      atomicAnd(&element, ~(1 << offset));
  }

  __device__ void atomicOrBitAt(const U bIndex, const bool value) {
    const U offset = bitOffset(bIndex);
    T &element = elementContainingBitAt(bIndex);
    if (value) // set bit to 1
      atomicOr(&element, (1 << offset));
    else // set bit to 0
      atomicAnd(&element, ~(1 << offset));
  }
}; // end struct Bitset

} // namespace containers

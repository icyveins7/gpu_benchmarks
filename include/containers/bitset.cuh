#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdexcept>

namespace containers {

/**
 * @brief A struct containing a single (unsigned) POD element representing bits.
 */
template <typename Tval = unsigned int, typename Tidx = int> struct Bitset {
  static_assert(std::is_unsigned<Tval>::value, "T must be unsigned");
  Tval value;

  /**
   * @brief Default constructor. Note that this does not default initialize the
   * internal value to 0. This is to prevent unwanted operations by threads in
   * the kernel (otherwise all threads may attempt to write to all values in an
   * array, resulting in race conditions).
   */
  __host__ __device__ Bitset() {}
  __host__ __device__ Bitset(Tval _value) : value(_value) {}

  __host__ __device__ static constexpr Tidx numBitsPerElement() {
    return (int)sizeof(Tval) * 8;
  }

  /**
   * @brief Creates a bit mask with only a 1 at the specified index.
   *
   * @param bIndex Bit index
   * @return Bit mask
   */
  __host__ __device__ static Tval bitmask(const Tidx bIndex) {
    return 1 << bIndex;
  }

  /**
   * @brief Creates a bit mask over a contiguous range of bits. Courtesy of
   * https://stackoverflow.com/questions/39321580/fastest-way-to-produce-a-mask-with-n-ones-starting-at-position-i.
   * Example: 0b00011110 -> (start = 1, len = 4)
   * Note that this will not produce an error if the range extends out of bounds
   * of the element, e.g. for Tval = uint8 if bIndexStart > 8 then the mask is
   * simply all zeroes.
   *
   * @param bIndexStart Starting bit index
   * @param len Number of bits set in the bitmask
   * @return Bit mask
   */
  __host__ __device__ static constexpr Tval bitmask(const Tidx bIndexStart,
                                                    const Tidx len) {
    return ((static_cast<Tval>(1) << len) - 1) << bIndexStart;
  }

  /**
   * @brief Helper function to check if the requested bit index exceeds the
   * number of bits stored. Onus is on the programmer to decide if this needs to
   * be called.
   *
   * @param bIndex Bit index
   * @return True if the bit index is valid
   */
  __host__ __device__ bool isValidBitIndex(const Tidx bIndex) const {
    return bIndex < numBitsPerElement();
  }

  /**
   * @brief Sets the bit at the specified index with the input value.
   *
   * @param bIndex Bit index
   * @param in Value to set, accepts truthy values like 1 or 0
   */
  __host__ __device__ void setBitAt(const Tidx bIndex, const bool in) {
    if (in) // set bit to 1
      value |= bitmask(bIndex);
    else // set bit to 0
      value &= ~bitmask(bIndex);
  }

  /**
   * @brief Returns the bit at the specified bit index.
   *
   * @param bIndex Bit index
   * @return True if the bit is set
   */
  __host__ __device__ bool getBitAt(const Tidx bIndex) const {
    return (value & bitmask(bIndex)) != 0;
  }

  /**
   * @brief Wrapper to get the value with a range bitmask.
   *
   * @param bIndexStart Starting bit index
   * @param len Number of bits set in the bitmask
   * @return Masked value
   */
  __host__ __device__ Tval get(const Tidx bIndexStart, const Tidx len) const {
    return (value & bitmask(bIndexStart, len));
  }

  /**
   * @brief Atomically sets the bit at the specified bit index.
   *
   * @param bIndex Bit index
   * @param in Value to set, accepts truthy values like 1 or 0
   */
  __device__ void setBitAtAtomically(const Tidx bIndex, const bool in) {
    if (in) // set bit to 1
      atomicOr(&value, bitmask(bIndex));
    else // set bit to 0
      atomicAnd(&value, ~bitmask(bIndex));
  }

  /**
   * @brief Strictly atomic ORs the bit at the specified bit index.
   *
   * @param bIndex Bit index
   * @param in Value to set, accepts truthy values like 1 or 0
   */
  __device__ void atomicOrBitAt(const Tidx bIndex, const bool in) {
    if (in)
      atomicOr(&value, bitmask(bIndex));
    // If 0, no need to do anything
  }
};

/**
 * @brief A container that represents a view over some data containing raw bits.
 */
template <typename Tval = unsigned int, typename Tidx = int>
struct BitsetArray {
  static_assert(std::is_unsigned<Tval>::value, "Tval must be unsigned");

  Bitset<Tval, Tidx> *data = nullptr;
  Tidx numDataElements = 0;
  Tidx numBits = 0;

  __host__ __device__ BitsetArray() {};
  __host__ __device__ BitsetArray(Bitset<Tval, Tidx> *_data,
                                  Tidx _numDataElements, Tidx _numBits)
      : data(_data), numDataElements(_numDataElements), numBits(_numBits) {
    // There is no easy way to check this inside the kernel since we cannot
    // throw, so it is expected that the numBits can fit inside numDataElements
  }
  __host__ __device__ BitsetArray(Bitset<Tval, Tidx> *_data, Tidx _numBits)
      : data(_data), numBits(_numBits) {
    // If only bits provided, assume the data elements were allocated using
    // numElementsRequiredFor().
    numDataElements = this->numElementsRequiredFor(_numBits);
  }
  // Host-only constructor from thrust::device_vector
  __host__ BitsetArray(thrust::device_vector<Bitset<Tval, Tidx>> &_data,
                       Tidx _numBits)
      : data(_data.data().get()), numDataElements(_data.size()),
        numBits(_numBits) {
    if (numBits > numDataElements * numBitsPerElement()) {
      throw std::invalid_argument(
          "numBits > numDataElements * numBitsPerElement()");
    }
  }
  // Host-only constructor from thrust::host_vector
  __host__ BitsetArray(thrust::host_vector<Bitset<Tval, Tidx>> &_data,
                       Tidx _numBits)
      : data(_data.data()), numDataElements(_data.size()), numBits(_numBits) {
    if (numBits > numDataElements * numBitsPerElement()) {
      throw std::invalid_argument(
          "numBits > numDataElements * numBitsPerElement()");
    }
  }

  /**
   * @brief Redirects to Bitset<Tval, Tidx>::numBitsPerElement().
   *
   * @return Number of bits per element
   */
  __host__ __device__ static constexpr Tidx numBitsPerElement() {
    return Bitset<Tval, Tidx>::numBitsPerElement();
  }

  /**
   * @brief Returns the number of elements required for a given number of bits,
   * for the current templated type. Useful to help in allocation size for the
   * templated type.
   *
   * @param numBits Total number of desired bits
   * @return Total number of required elements of type T
   */
  __host__ __device__ static Tidx numElementsRequiredFor(const Tidx numBits) {
    Tidx numElements = numBits / numBitsPerElement();
    if (numBits % numBitsPerElement() > 0)
      numElements++;
    return numElements;
  }

  /**
   * @brief Returns true if the number of bits is valid.
   */
  __host__ __device__ bool hasValidNumBits() const {
    return numBits <= numDataElements * numBitsPerElement();
  }

  // ===================================================================
  // ====================== ELEMENT MANIPULATION =======================
  // ===================================================================

  /**
   * @brief Checks if the provided element index is in-range.
   *
   * @param index Element index
   * @return True if the index is valid
   */
  __host__ __device__ bool isValidElementIndex(const Tidx index) const {
    return index < numDataElements;
  }

  /**
   * @brief Returns a reference to the element at the specified index.
   *
   * @param index Element index
   * @return Bitset element reference
   */
  __host__ __device__ Bitset<Tval, Tidx> &elementAt(const Tidx index) {
    return data[index];
  }
  /**
   * @brief Returns a const reference to the element at the specified index.
   *
   * @param index Element index
   * @return Const bitset element reference
   */
  __host__ __device__ const Bitset<Tval, Tidx> &
  elementAt(const Tidx index) const {
    return data[index];
  }

  /**
   * @brief Returns the element containing the bit index. See bitOffset() to
   * retrieve the bit offset within the element.
   *
   * @param bIndex (Array-wide) Bit index
   * @return Element reference
   */
  __host__ __device__ Bitset<Tval, Tidx> &
  elementContainingBitAt(const Tidx bIndex) {
    return elementAt(bIndex / numBitsPerElement());
  }

  /**
   * @brief Returns the element containing the bit index. See bitOffset() to
   * retrieve the bit offset within the element.
   *
   * @param bIndex (Array-wide) Bit index
   * @return Const element reference
   */
  __host__ __device__ const Bitset<Tval, Tidx> &
  elementContainingBitAt(const Tidx bIndex) const {
    return elementAt(bIndex / numBitsPerElement());
  }

  // ===================================================================
  // ========================== BIT MANIPULATION =======================
  // ===================================================================

  // /**
  //  * @brief Returns the element index containing the bit at the specified bit
  //  * index.
  //  *
  //  * @param bIndex (Array-wide) Bit index
  //  * @return Element index
  //  */
  // __host__ __device__ Tidx elementOffset(const Tidx bIndex) const {
  //   return bIndex / numBitsPerElement();
  // }

  __host__ __device__ bool isValidBitIndex(const Tidx bIndex) const {
    return bIndex < numBits;
  }

  /**
   * @brief Returns the number of bits offset inside the element containing the
   bit index. See elementContainingBitAt() to retrieve the element itself.

   * @param bIndex (Array-wide) Bit index
   * @return Number of bits offset
   */
  __host__ __device__ Tidx bitOffset(const Tidx bIndex) const {
    return bIndex % numBitsPerElement();
  }

  /**
   * @brief Primary getter for bits.
   *
   * @param bIndex (Array-wide) Bit index
   * @return True if the bit is set
   */
  __host__ __device__ bool getBitAt(const Tidx bIndex) const {
    const Tidx offset = bitOffset(bIndex);
    const Bitset<Tval, Tidx> &element = elementContainingBitAt(bIndex);
    return element.getBitAt(offset);
  }

  /**
   * @brief Primary setter for bits.
   *
   * @param bIndex (Array-wide) Bit index
   * @param value Value to set. Using truthy-values like 1 or 0 directly also
   * works.
   */
  __host__ __device__ void setBitAt(const Tidx bIndex, const bool value) {
    const Tidx offset = bitOffset(bIndex);
    Bitset<Tval, Tidx> &element = elementContainingBitAt(bIndex);
    element.setBitAt(offset, value);
  }

  // ========================================================================
  // ========================================================================
  // ========================== DEVICE-ONLY =================================
  // ========================================================================
  // ========================================================================

  /**
   * @brief Setter for bits atomically. Use this if multiple threads will write
   * to the same element.
   *
   * @param bIndex Bit index
   * @param value Value to set. Using truthy-values like 1 or 0 directly also
   * works.
   */
  __device__ void setBitAtAtomically(const Tidx bIndex, const bool value) {
    const Tidx offset = bitOffset(bIndex);
    Bitset<Tval, Tidx> &element = elementContainingBitAt(bIndex);
    element.setBitAtAtomically(offset, value);
  }

  __device__ void atomicOrBitAt(const Tidx bIndex, const bool value) {
    const Tidx offset = bitOffset(bIndex);
    Bitset<Tval, Tidx> &element = elementContainingBitAt(bIndex);
    element.atomicOrBitAt(offset, value);
  }

  /**
   * @brief Block-wide reduction to find minimum (first) set bit index across
   * array. Programmer is responsible for performing __syncthreads() after this.
   *
   * @param minIdx Atomically updated index pointer
   */
  __device__ void argminBit(Tidx *minIdx) const;

  /**
   * @brief Block-wide reduction to find first minimum set bit index after a
   * floor. Intended to be used so that one can iterate through set bits in the
   * array without explicitly zero-ing out previous bits.
   *
   * @param minIdx Atomically updated index pointer
   * @param floor Floor i.e. previous minimum bit index
   */
  // __device__ void argminBit(Tidx *minIdx, const Tidx floor) const;
}; // end struct Bitset

// Specializations, because we use intrinsics for specific types

template <>
__device__ void BitsetArray<unsigned int, int>::argminBit(int *minIdx) const {
  // Initialize using first thread
  if (threadIdx.x == 0 && threadIdx.y == 0)
    *minIdx = numBits;
  __syncthreads();

  int minBit = numBits;
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < numDataElements;
       i += blockDim.y * blockDim.x) {
    const Bitset<unsigned int, int> &elem = elementAt(i);
    if (elem.value != 0) {
      // ffs returns 1 for 0b000....001, 2 for 0b000....010, etc
      // hence we want to -1 from this value since our convention starts from 0
      int localMinBit = __ffs(*reinterpret_cast<const int *>(&elem.value)) - 1;
      // printf("Elem: %08X, localMinBit = %d, getBit = %d\n", elem.value,
      //        localMinBit, elem.getBitAt(localMinBit));
      minBit = min(minBit, localMinBit + i * 32);
      break;
      // TODO: i should probably warp reduce this and have the entire warp break
      // too, but for now this should be ok since we have a short array
    }
  }
  // printf("Thread %d: minBit = %d\n", threadIdx.y * blockDim.y + threadIdx.x,
  //        minBit);
  atomicMin(minIdx, minBit);
}

// template <>
// __device__ void
// BitsetArray<unsigned int, int>::argminBit(int *minIdx, const int floor) const
// {
//   // Initialize using first thread
//   if (threadIdx.x == 0 && threadIdx.y == 0)
//     *minIdx = numBits;
//   __syncthreads();
//
//   int minBit = numBits;
//   for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < numDataElements;
//        i += blockDim.y * blockDim.x) {
//     const Bitset<unsigned int, int> &elem = elementAt(i);
//     if (elem.value != 0) {
//       // ffs returns 1 for 0b000....001, 2 for 0b000....010, etc
//       // hence we want to -1 from this value since our convention starts from
//       0 int localMinBit = __ffs(*reinterpret_cast<const int *>(&elem.value))
//       - 1;
//       // printf("Elem: %08X, localMinBit = %d, getBit = %d\n", elem.value,
//       //        localMinBit, elem.getBitAt(localMinBit));
//       minBit = min(minBit, localMinBit + i * 32);
//       break;
//       // TODO: i should probably warp reduce this and have the entire warp
//       break
//       // too, but for now this should be ok since we have a short array
//     }
//   }
//   // printf("Thread %d: minBit = %d\n", threadIdx.y * blockDim.y +
//   threadIdx.x,
//   //        minBit);
//   atomicMin(minIdx, minBit);
// }

} // namespace containers

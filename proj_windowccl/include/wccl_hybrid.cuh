/*
In wccl_kernels.cuh, we defined many local intra-tile -> global inter-tile
kernels that could be used to perform the WCCL task. This was primarily
motivated by the sequential nature of WCCL, and hence the need for
synchronization and/or atomics.

The only viable method for the global inter-tile kernel (or the best so far) has
been to perform union-find. This however is extremely expensive when the window
distance is large, and dominates the duration in the best performing pipelines
(e.g. local neighbour chainer -> global union find), with up to 80-90% of the
time spent in this. At this point, it is prudent to study how much this can be
mitigated by doing *some* work in the CPU, transferring minimal amounts for
synchronization while performing most of the parallelizable work in the GPU.

For the hybrid neighbour chainer class, we will execute some parts on the CPU
and some parts on the GPU. At some points, we will need to update an unknown
number of values on the GPU (during beta consumption for example). Whether to do
this in a small loop over each index, or to simply update the entire array,
depends on the number of elements being updated. In initial timing tests using
the bitsets, it seems like for 8M elements we'll be sticking to about 100
elements before we update the entire array instead. Note that these are
*elements* not the individual bits!
*/

#pragma once

#include "containers/bitset.cuh"
#include "pinnedalloc.cuh"

#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace wccl {

template <typename T> __host__ __device__ constexpr int numBitsPerElement() {
  return sizeof(T) * 8;
}

class HybridNeighbourChainer {
public:
  HybridNeighbourChainer(const int rows, const int cols);

  void execute(const int seedIndex, const int2 windowDist);

  void fillBeta(const uint8_t *input);

  unsigned int getNextBeta();
  inline void consumeBeta(const int idx) {
    int elementIdx = idx / numBitsPerElement<unsigned int>();
    int offset = idx % numBitsPerElement<unsigned int>();
    h_beta[elementIdx] ^= (1 << offset);
    d_beta[elementIdx] ^= (1 << offset);
  }

  inline int getRows() const { return m_rows; }
  inline int getCols() const { return m_cols; }
  inline int getColElements() const { return m_colElements; }
  inline int getPaddedCols() const {
    return m_colElements * numBitsPerElement<unsigned int>();
  }
  inline thrust::pinned_host_vector<unsigned int> &getHostBeta() {
    return h_beta;
  }
  inline thrust::device_vector<unsigned int> &getDeviceBeta() { return d_beta; }

protected:
  int m_rows;        // true number of rows (bits)
  int m_cols;        // true number of cols (bits)
  int m_colElements; // number of elements used to represent the columns (padded
                     // to nearest int)
  int m_lastBetaIdx = 0;

  thrust::pinned_host_vector<unsigned int> h_beta;
  thrust::device_vector<unsigned int> d_beta;
  thrust::device_vector<unsigned int> d_gamma;
  thrust::pinned_host_vector<unsigned int> h_gamma;
  thrust::device_vector<unsigned int> d_seedrow;
  thrust::device_vector<unsigned int> d_gammaIdx;
  thrust::pinned_host_vector<unsigned int> h_gammaIdx;
};

} // namespace wccl

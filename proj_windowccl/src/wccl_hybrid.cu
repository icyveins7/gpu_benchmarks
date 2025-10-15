#include "wccl.h"
#include "wccl_hybrid.cuh"
#include <nvtx3/nvToolsExt.h>

namespace wccl {

__device__ unsigned int
compute_masked_element(const int leftWindowBit, const int rightWindowBit,
                       const int row, const int colElement,
                       const unsigned int initialElement, const int targetRow,
                       const int2 windowDist) {
  // Determine element coordinates
  int leftBit = colElement * numBitsPerElement<unsigned int>();
  int rightBit = leftBit + numBitsPerElement<unsigned int>();

  // Compute mask
  unsigned int elementMask = 0;
  if (row >= targetRow - windowDist.y && row <= targetRow + windowDist.y &&
      leftWindowBit <= rightBit && rightWindowBit >= leftBit) {

    // int leftBound = max(leftWindowBit, leftBit);
    // int rightBound = min(rightWindowBit, rightBit);
    // int maskLength = rightBound - leftBound + 1;
    // if (maskLength == numBitsPerElement<unsigned int>()) {
    //   elementMask = 0xFFFFFFFF;
    // } else {
    //   elementMask = ((static_cast<unsigned int>(1) << maskLength) - 1)
    //                 << (leftBound - leftBit);
    // }

    for (int j = 0; j < numBitsPerElement<unsigned int>(); j++) {
      if (leftBit + j >= leftWindowBit && leftBit + j <= rightWindowBit) {
        elementMask |= (1 << j);
      }
    }
  }
  return initialElement & elementMask;
}

__global__ void construct_seedrow_kernel(const unsigned int *beta,
                                         unsigned int *seedrow, const int rows,
                                         const int colElements,
                                         const int2 windowDist,
                                         const int seedIndex) {
  // Compute effective row and column bit index from seed
  const int seedRow =
      seedIndex / (colElements * numBitsPerElement<unsigned int>());
  const int seedCol =
      seedIndex % (colElements * numBitsPerElement<unsigned int>());
  const int leftWindowBit = seedCol - windowDist.x;
  const int rightWindowBit = seedCol + windowDist.x;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < rows * colElements;
       i += gridDim.x * blockDim.x) {
    int row = i / colElements;
    int colElement = i % colElements;
    unsigned int out =
        compute_masked_element(leftWindowBit, rightWindowBit, row, colElement,
                               beta[i], seedRow, windowDist);
    seedrow[i] = out;
  }
}

__global__ void
merge_gammas_to_seedrow_kernel(const ushort2 *gammaIdx, const int numGammaIdx,
                               unsigned int *seedrow, const int rows,
                               const int colElements, const int2 windowDist,
                               const unsigned int *beta) {

  for (int i = blockDim.y * blockIdx.y + threadIdx.y; i < rows;
       i += gridDim.y * blockDim.y) {
    for (int j = blockDim.x * blockIdx.x + threadIdx.x; j < colElements;
         j += gridDim.x * blockDim.x) {
      // Define output for current element
      int row = i;
      int colElement = j;

      unsigned int mask = 0;
      for (int g = 0; g < numGammaIdx; ++g) {
        // // DEBUG
        // if (threadIdx.x == 0)
        //   printf("gammaIdx[%d] = %d\n", g, gammaIdx[g]);
        // Compute effective row and column bit index from seed
        const int targetRow = gammaIdx[g].y;
        const int targetCol = gammaIdx[g].x;
        // gammaIdx[g] / (colElements * numBitsPerElement<unsigned int>());
        // const int targetCol =
        //     gammaIdx[g] % (colElements * numBitsPerElement<unsigned int>());
        const int leftWindowBit = targetCol - windowDist.x;
        const int rightWindowBit = targetCol + windowDist.x;

        mask |= compute_masked_element(leftWindowBit, rightWindowBit, row,
                                       colElement, beta[i * colElements + j],
                                       targetRow, windowDist);
      }

      if (mask != 0)
        seedrow[i * colElements + j] = seedrow[i * colElements + j] | mask;
    }
  }
}

HybridNeighbourChainer::HybridNeighbourChainer(const int rows, const int cols)
    : m_rows(rows), m_cols(cols),
      m_colElements((cols % 32 == 0)
                        ? cols / 32
                        : cols / 32 + 1), // we pad along the columns
      h_beta(m_colElements * m_rows), d_beta(m_colElements * m_rows),
      d_gamma(m_colElements * m_rows), h_gamma(m_colElements * m_rows),
      d_seedrow(m_colElements * m_rows), d_gammaIdx(m_cols * m_rows),
      h_gammaIdx(m_cols * m_rows) {}

void HybridNeighbourChainer::fillBeta(const uint8_t *input) {
  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      h_beta[i * m_colElements + j / 32] |= (input[i * m_cols + j] << (j % 32));
    }
  }

  thrust::copy(h_beta.begin(), h_beta.end(), d_beta.begin());
}

void HybridNeighbourChainer::execute(const int seedIndex,
                                     const int2 windowDist) {
  // 1. Construct seed row [GPU]
  {
    // NOTE: using just 512 since thrust calls seem to use this,
    // probably to ensure each block can hide latencies
    int tpb = 256;
    int blks = 512; //(m_rows * m_colElements + tpb - 1) / tpb;
    construct_seedrow_kernel<<<blks, tpb>>>(
        d_beta.data().get(), d_seedrow.data().get(), m_rows, m_colElements,
        windowDist, seedIndex);
  }
  // // DEBUG
  // thrust::host_vector<unsigned int> h_seedrow = d_seedrow;
  // for (int i = 0; i < (int)h_seedrow.size(); ++i) {
  //   printf("SEEDROW: %X\n", h_seedrow[i]);
  // }

  // 2. Consume beta index [GPU]
  this->consumeBeta(seedIndex);
  // // DEBUG
  // printf("BETA AFTER INITIAL CONSUMPTION\n%s\n",
  //        wccl::bitstring(&h_beta[0], m_rows, m_colElements).c_str());

  bool continueIterations = true;
  while (continueIterations) {
    // 3. Compute gamma (neighbours of interest) [GPU]
    nvtxRangePushA("compute gamma via bit_and");
    thrust::transform(d_seedrow.begin(), d_seedrow.end(), d_beta.begin(),
                      d_gamma.begin(), thrust::bit_and<unsigned int>());
    nvtxRangePop();

    // 4. Copy gamma back to host
    thrust::copy(d_gamma.begin(), d_gamma.end(), h_gamma.begin());
    // // DEBUG
    // printf("GAMMA\n%s\n",
    //        wccl::bitstring(&h_gamma[0], m_rows, m_colElements).c_str());

    // 5. Compute gamma indices in CPU; if gamma empty, end (go to 9)
    nvtxRangePushA("compute gamma indices CPU");
    // h_gammaIdx.clear(); // thrust pinned host vector is still calling kernels
    // when using pushback?
    unsigned int gammaCounter = 0;

    for (int i = 0; i < (int)h_gamma.size(); ++i) {
      if (h_gamma[i] != 0) {
        for (int j = 0; j < numBitsPerElement<unsigned int>(); ++j) {
          if (h_gamma[i] & (1 << j)) {
            h_gammaIdx[gammaCounter].x =
                (i % m_colElements) * numBitsPerElement<unsigned int>() + j;
            h_gammaIdx[gammaCounter].y = i / m_colElements;
            // i *numBitsPerElement<unsigned int>() + j;
            gammaCounter++;
            // h_gammaIdx.push_back(i * numBitsPerElement<unsigned int>() + j);
          }
        }
      }
    }
    // if (h_gammaIdx.size() == 0) {
    if (gammaCounter == 0) {
      continueIterations = false;
      break;
    }
    nvtxRangePop();
    // DEBUG:
    printf("gammaIdx size: %u\n", gammaCounter);
    // If gamma not empty, continue

    // -  6. Copy gamma index list to device
    thrust::copy(h_gammaIdx.begin(), h_gammaIdx.begin() + gammaCounter,
                 d_gammaIdx.begin());

    // -  7. Merge (bitwise OR) all gamma index rows with seed row [GPU]
    {
      char msg[32];
      snprintf(msg, sizeof(msg), "mergekernel[%u]", gammaCounter);
      nvtxRangePushA(msg);

      dim3 tpb(32, 2);
      // int blks = (m_rows * m_colElements + tpb - 1) / tpb;
      dim3 blks(m_colElements / tpb.x + (m_colElements % tpb.x > 0 ? 1 : 0),
                m_rows / tpb.y + (m_rows % tpb.y > 0 ? 1 : 0));
      merge_gammas_to_seedrow_kernel<<<blks, tpb>>>(
          d_gammaIdx.data().get(),
          (int)gammaCounter, // use gammaCounter directly
          d_seedrow.data().get(), m_rows, m_colElements, windowDist,
          d_beta.data().get());
      nvtxRangePop();
    }
    // // DEBUG
    // h_seedrow = d_seedrow;
    // printf("SEEDROW AT END\n%s\n",
    //        wccl::bitstring(&h_seedrow[0], m_rows, m_colElements).c_str());

    // Consume all gammas before continuing
    // printf("Consuming gammas \n");
    nvtxRangePushA("bitxor gammas to beta");
    thrust::transform(d_beta.begin(), d_beta.end(), d_gamma.begin(),
                      d_beta.begin(), thrust::bit_xor<unsigned int>());
    nvtxRangePop();

    // // DEBUG
    // h_beta = d_beta;
    // printf("BETA AT END\n%s\n",
    //        wccl::bitstring(&h_beta[0], m_rows, m_colElements).c_str());
    // -  8. Go to 3.
  }

  // 9. Once done, copy seed row back to CPU

  // 10. Readout.

  // 11. Re-copy beta back to host for tracking
  h_beta = d_beta;
}

unsigned int HybridNeighbourChainer::getNextBeta() {
  for (int i = m_lastBetaIdx / 32; i < (int)h_beta.size(); ++i) {
    if (h_beta[i] != 0) {
      for (int j = 0; j < numBitsPerElement<unsigned int>(); ++j) {
        if (h_beta[i] & (1 << j)) {
          m_lastBetaIdx = i * numBitsPerElement<unsigned int>() + j;
          printf("NEXT BETA: %d\n", m_lastBetaIdx);
          return m_lastBetaIdx;
        }
      }
    }
  }
  printf("No more beta\n");
  return m_rows * m_colElements * numBitsPerElement<unsigned int>();
}

} // end namespace wccl

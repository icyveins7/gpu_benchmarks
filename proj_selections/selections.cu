#include "selectors.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

int main() {
  printf("Selection kernel tests\n==============\n");

  // Create some input
  unsigned int iRows = 128, iCols = 128;
  thrust::host_vector<int> h_input(iRows * iCols);
  thrust::sequence(h_input.begin(), h_input.end());

  thrust::device_vector<int> d_input = h_input;

  // Allocate output
  unsigned int oRows = 2, oCols = 2, oMaxLength = 128;
  constexpr int MAX_SLICES = 2;
  thrust::device_vector<int> d_output(oRows * oCols * oMaxLength);
  thrust::device_vector<unsigned int> d_outputLengths(oRows * oCols);

  // Create some slices
  thrust::host_vector<SliceBounds<unsigned int>> h_sliceIdx(oRows * oCols *
                                                            MAX_SLICES);
  thrust::host_vector<unsigned int> h_numSlices(oRows * oCols);

  // ===
  h_sliceIdx[0 * MAX_SLICES + 0] = {5 * iCols + 0,
                                    5 * iCols + 33}; // 33 elements
  h_sliceIdx[0 * MAX_SLICES + 1] = {6 * iCols + 15,
                                    6 * iCols + 32}; // 18 elements
  h_numSlices[0] = 2;

  // ===
  h_sliceIdx[1 * MAX_SLICES + 0] = {51 * iCols + 8,
                                    51 * iCols + 11}; // 4 elements
  h_numSlices[1] = 1;

  // ===
  h_sliceIdx[2 * MAX_SLICES + 0] = {77 * iCols + 64,
                                    77 * iCols + 80}; // 17 elements
  h_numSlices[2] = 1;

  // ===
  h_sliceIdx[3 * MAX_SLICES + 0] = {97 * iCols + 64,
                                    97 * iCols + 80}; // 17 elements
  h_sliceIdx[3 * MAX_SLICES + 1] = {98 * iCols + 64,
                                    98 * iCols + 80}; // 17 elements
  h_numSlices[3] = 2;

  thrust::device_vector<SliceBounds<unsigned int>> d_sliceIdx = h_sliceIdx;
  thrust::device_vector<unsigned int> d_numSlices = h_numSlices;

  // Call the kernel
  blockwise_select_1d_slices_kernel<int, unsigned int, MAX_SLICES>
      <<<dim3(oCols * oRows, 1, 1), dim3(32, 1, 1)>>>(
          thrust::raw_pointer_cast(d_input.data()), iRows * iCols,
          thrust::raw_pointer_cast(d_output.data()),
          thrust::raw_pointer_cast(d_outputLengths.data()), oRows * oCols,
          oMaxLength, thrust::raw_pointer_cast(d_sliceIdx.data()),
          thrust::raw_pointer_cast(d_numSlices.data()));

  // Read the output
  thrust::host_vector<int> h_output = d_output;
  thrust::host_vector<unsigned int> h_outputLengths = d_outputLengths;
  for (int oRow = 0; oRow < (int)oRows; ++oRow) {
    for (int oCol = 0; oCol < (int)oCols; ++oCol) {
      printf("===================\noRow=%d, oCol=%d\n", oRow, oCol);
      unsigned int length = h_outputLengths[oRow * oCols + oCol];
      printf("Total elements: %u\n", length);
      for (unsigned int i = 0; i < length; ++i) {
        std::cout << h_output[oRow * oCols * oMaxLength + oCol * oMaxLength + i]
                  << " ";
      }
      std::cout << std::endl;
    }
  }

  // Try kernel again with some downsampling
  printf("\n================ Downsampling =============\n");
  thrust::device_vector<unsigned char> d_skipsPerSlice(oRows * oCols);
  // Set all to 2
  thrust::fill(d_skipsPerSlice.begin(), d_skipsPerSlice.end(), 10);

  blockwise_select_1d_slices_kernel<int, unsigned int, MAX_SLICES,
                                    unsigned char>
      <<<dim3(oCols * oRows, 1, 1), dim3(32, 1, 1)>>>(
          thrust::raw_pointer_cast(d_input.data()), iRows * iCols,
          thrust::raw_pointer_cast(d_output.data()),
          thrust::raw_pointer_cast(d_outputLengths.data()), oRows * oCols,
          oMaxLength, thrust::raw_pointer_cast(d_sliceIdx.data()),
          thrust::raw_pointer_cast(d_numSlices.data()),
          thrust::raw_pointer_cast(d_skipsPerSlice.data()));

  // Read again
  h_output = d_output;
  h_outputLengths = d_outputLengths;
  for (int oRow = 0; oRow < (int)oRows; ++oRow) {
    for (int oCol = 0; oCol < (int)oCols; ++oCol) {
      printf("===================\noRow=%d, oCol=%d\n", oRow, oCol);
      unsigned int length = h_outputLengths[oRow * oCols + oCol];
      printf("Total elements: %u\n", length);
      for (unsigned int i = 0; i < length; ++i) {
        std::cout << h_output[oRow * oCols * oMaxLength + oCol * oMaxLength + i]
                  << " ";
      }
      std::cout << std::endl;
    }
  }

  return 0;
}

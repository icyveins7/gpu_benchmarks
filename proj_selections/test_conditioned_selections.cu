#include "selectors.cuh"

#include <cstdlib>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

TEST(ConditionedDownsampling, MaximumValidator) {
  int rows = 64;
  int cols = 64;
  thrust::host_vector<unsigned short> h_input(rows * cols);
  thrust::host_vector<unsigned short> h_conditional(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      h_conditional[i * cols + j] = std::rand() % 10;
      h_input[i * cols + j] = std::rand() % 10;
    }
  }
  thrust::device_vector<unsigned short> d_input = h_input;
  thrust::device_vector<unsigned short> d_conditional = h_conditional;

  // Allocate sufficient output for ROI
  const unsigned int roiStartRow = 5, roiStartCol = 5;
  const unsigned int roiNumRows = 32, roiNumCols = 32;
  const unsigned int roiRowStride = 1, roiColStride = 2;
  // Count the number of rows/cols actually used
  const unsigned int roiNumStridedRows =
      roiNumRows / roiRowStride + (roiNumRows % roiRowStride == 0 ? 0 : 1);
  const unsigned int roiNumStridedCols =
      roiNumCols / roiColStride + (roiNumCols % roiColStride == 0 ? 0 : 1);
  thrust::device_vector<unsigned short> d_out(roiNumStridedRows *
                                              roiNumStridedCols);
  printf("roiNumStridedRows = %u, roiNumStridedCols = %u, d_out size = %zd\n",
         roiNumStridedRows, roiNumStridedCols, d_out.size());

  // Create a validator
  MaximumValidator<unsigned short> maxValidator{7};

  dim3 NUM_THREADS(32, 8);
  dim3 NUM_BLKS(roiNumCols / NUM_THREADS.x + 1, roiNumRows / NUM_THREADS.y + 1);
  conditioned_downsampling_kernel<unsigned short><<<NUM_BLKS, NUM_THREADS>>>(
      d_input.data().get(), d_conditional.data().get(), rows, cols, roiStartRow,
      roiStartCol, roiNumRows, roiNumCols, roiRowStride, roiColStride,
      d_out.data().get(), roiNumStridedRows, roiNumStridedCols, maxValidator);

  thrust::host_vector<unsigned short> h_out = d_out;

  for (unsigned int i = 0; i < roiNumRows; i += roiRowStride) {
    for (unsigned int j = 0; j < roiNumCols; j += roiColStride) {
      int inputIdx = (i + roiStartRow) * cols + (j + roiStartCol);
      int outputIdx =
          (i / roiRowStride) * roiNumStridedCols + (j / roiColStride);
      if (h_conditional[inputIdx] < maxValidator.maxVal) {
        EXPECT_EQ(h_out[outputIdx], h_input[inputIdx]);
      } else {
        EXPECT_EQ(h_out[outputIdx], std::numeric_limits<unsigned short>::max());
      }
    }
  }
}

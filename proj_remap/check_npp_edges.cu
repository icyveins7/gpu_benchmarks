#include <iostream>
#include <npp.h>
#include <random>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

int main(int argc, char *argv[]) {
  // We create a simple 2x2 example
  // The coordinates are expected to be (0,0), (0,1), (1,0), (1,1)
  int srcWidth = 2;
  int srcHeight = 2;
  thrust::device_vector<float> d_srcImg(srcWidth * srcHeight);
  // We set all values to be above 1 so we can easily compare later
  thrust::sequence(d_srcImg.begin(), d_srcImg.end(), 1.0f);

  // Create the requested pixel positions
  // We extend beyond the edges of the image to see what happens
  // This means we go from (-2, -2) to (3, 3), a border of 2 pixels all around
  // We place our resolution at 0.2f to roughly look at the transition
  // This is a total of 26 points in both directions
  const size_t reqLen = 26;
  const float step = 0.2f;

  thrust::device_vector<float> d_x(reqLen * reqLen);
  thrust::device_vector<float> d_y(reqLen * reqLen);

  // Fill our points for x, which is just the same for every row
  for (int i = 0; i < reqLen; ++i)
    // Generate 0 to 50
    thrust::sequence(d_x.begin() + i * reqLen, d_x.begin() + (i + 1) * reqLen,
                     0.0f);
  // Multiply by step (0.1f) and add to start (-2.0f)
  thrust::transform(d_x.begin(), d_x.end(), d_x.begin(),
                    step * thrust::placeholders::_1);
  thrust::transform(d_x.begin(), d_x.end(), d_x.begin(),
                    -2.0f + thrust::placeholders::_1);

  // Fill our points for y, which is the same for every column
  for (int i = 0; i < reqLen; ++i)
    thrust::fill(d_y.begin() + i * reqLen, d_y.begin() + (i + 1) * reqLen,
                 (float)(-2.0f + i * step));

  // Let's print them to check
  thrust::host_vector<float> h_x = d_x;
  thrust::host_vector<float> h_y = d_y;

  printf("x: \n");
  for (int i = 0; i < reqLen; ++i) {
    for (int j = 0; j < reqLen; ++j) {
      printf("%.1f, ", h_x[i * reqLen + j]);
    }
    printf("\n");
  }

  printf("y: \n");
  for (int i = 0; i < reqLen; ++i) {
    for (int j = 0; j < reqLen; ++j) {
      printf("%.1f, ", h_y[i * reqLen + j]);
    }
    printf("\n");
  }

  // Now let's run the NPP remap
  NppiSize srcSize = {(int)srcWidth, (int)srcHeight};
  NppiRect srcROI = {0, 0, (int)srcWidth, (int)srcHeight};
  NppiSize dstSize = {(int)reqLen, (int)reqLen};
  thrust::device_vector<float> d_out(h_x.size());
  // Before we run it, set everything to a large value so it's obvious if it's
  // not being modified
  thrust::fill(d_out.begin(), d_out.end(), 9.9f);

  NppStatus status = nppiRemap_32f_C1R(
      thrust::raw_pointer_cast(d_srcImg.data()), srcSize,
      srcWidth * sizeof(float), srcROI, thrust::raw_pointer_cast(d_x.data()),
      reqLen * sizeof(float), thrust::raw_pointer_cast(d_y.data()),
      reqLen * sizeof(float), thrust::raw_pointer_cast(d_out.data()),
      reqLen * sizeof(float), dstSize, NPPI_INTER_LINEAR);

  // Print the output
  thrust::host_vector<float> h_out = d_out;
  printf("Out: \n");
  for (int i = 0; i < reqLen; ++i) {
    for (int j = 0; j < reqLen; ++j) {
      printf("%.1f, ", h_out[i * reqLen + j]);
    }
    printf("\n");
  }

  // A bit hard to see, so let's just print the coordinates of where it changes
  for (int i = 0; i < reqLen; ++i) {
    for (int j = 0; j < reqLen; ++j) {
      if (h_out[i * reqLen + j] != 9.9f) {
        printf("x: %.1f, y: %.1f\n", h_x[i * reqLen + j], h_y[i * reqLen + j]);
      }
    }
  }

  return 0;
}

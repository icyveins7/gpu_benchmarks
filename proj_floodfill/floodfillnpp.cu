#include <iostream>
#include <limits>
#include <npp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char *argv[]) {
  printf("Flood fill using NPP\n");

  using Tdata = Npp32u;

  int width = 100, height = 100;
  if (argc > 1) {
    width = std::atoi(argv[1]);
    height = std::atoi(argv[1]);
  }
  NppiSize outputSizeROI{width, height};
  printf("Size: %d x %d\n", width, height);

  thrust::host_vector<Tdata> h_img(width * height);
  NppiPoint centre{width / 2, height / 2};
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if ((i - centre.y) * (i - centre.y) + (j - centre.x) * (j - centre.x) <
          (width / 2) * (width / 2)) {
        h_img[i * width + j] = 1;
      }
      // some 'invalid pixels'
      if (std::rand() % 10 >= 9)
        h_img[i * width + j] = std::numeric_limits<Tdata>::max();
    }
  }
  // Ensure centre pixel is valid
  h_img[centre.y * width + centre.x] = 1;
  thrust::device_vector<Tdata> d_img = h_img;

  int bufSize = 0;
  nppiFloodFillGetBufferSize(outputSizeROI, &bufSize);
  printf("NPP buffer size: %d\n", bufSize);
  thrust::device_vector<Npp8u> nppBuffer(bufSize);

  nppiFloodFill_32u_C1IR(d_img.data().get(), width * sizeof(Tdata), centre, 2,
                         nppiNormInf, outputSizeROI, nullptr,
                         nppBuffer.data().get());

  h_img = d_img;

  if ((width <= 100) && (height <= 100)) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        char c;
        if (h_img[i * width + j] == std::numeric_limits<Tdata>::max()) {
          c = 'X';
        } else if (h_img[i * width + j] == 0) {
          c = '-';
        } else {
          c = '0' + h_img[i * width + j];
        }
        printf("%c ", c);
      }
      printf("\n");
    }
    printf("\n");
  }

  return 0;
}

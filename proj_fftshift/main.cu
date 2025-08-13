#include "fftshift.cuh"
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

int main() {
  {
    size_t width = 3, height = 3;
    thrust::host_vector<int> h_data(width * height);
    thrust::sequence(h_data.begin(), h_data.end());

    thrust::device_vector<int> d_data = h_data;
    thrust::device_vector<int> d_out(d_data.size());

    fftShift2D(thrust::raw_pointer_cast(d_data.data()),
               thrust::raw_pointer_cast(d_out.data()), width, height);

    h_data = d_out;

    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        printf("%2d ", h_data[y * width + x]);
      }
      std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
  }

  {
    size_t width = 4, height = 4;
    thrust::host_vector<int> h_data(width * height);
    thrust::sequence(h_data.begin(), h_data.end());

    thrust::device_vector<int> d_data = h_data;
    thrust::device_vector<int> d_out(d_data.size());

    fftShift2D(thrust::raw_pointer_cast(d_data.data()),
               thrust::raw_pointer_cast(d_out.data()), width, height);

    h_data = d_out;

    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        printf("%2d ", h_data[y * width + x]);
      }
      std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
  }

  {
    // specifically test with more than 1 block which is standardized to 16x16
    size_t width = 17, height = 17;
    thrust::host_vector<int> h_data(width * height);
    thrust::sequence(h_data.begin(), h_data.end());

    thrust::device_vector<int> d_data = h_data;
    thrust::device_vector<int> d_out(d_data.size());

    fftShift2D(thrust::raw_pointer_cast(d_data.data()),
               thrust::raw_pointer_cast(d_out.data()), width, height);

    h_data = d_out;

    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        printf("%3d ", h_data[y * width + x]);
      }
      std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
  }

  {
    // large one for timing
    size_t width = 8192, height = 8192;
    thrust::host_vector<thrust::complex<float>> h_data(width * height);
    thrust::sequence(h_data.begin(), h_data.end());

    thrust::device_vector<thrust::complex<float>> d_data = h_data;
    thrust::device_vector<float> d_out(d_data.size());

    fftShift2D<thrust::complex<float>, float, true>(
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_out.data()), width, height, {32, 4});

    h_data = d_out;

    std::cout << "=======================" << std::endl;
  }

  return 0;
}
